import torch
from torch import nn
import os
import argparse
import torch.distributed
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.l1=nn.Linear(1000,1000)
        self.bn=nn.BatchNorm1d(1000)
        self.l2=nn.Linear(1000,1)
        for p in self.l1.parameters():
            nn.init.ones_(p)
        for p in self.l2.parameters():
            nn.init.ones_(p)
    def forward(self,x):
        return self.l2(self.bn(self.l1(x)))

class dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(dataset,self).__init__()
        self.data = torch.eye(1000,1000)
    
    def __getitem__(self,index):
        return self.data[index],index
    def __len__(self):
        return len(self.data)

def train(args):
    # 进程组初始化
    torch.distributed.init_process_group(backend='tcp') # world_size 和 rank 此时直接从进程的环境变量中获取
     
    # 根据 local_rank 确定当前进程使用的GPU
    # local_rank=args.local_rank
    local_rank=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local_rank)
    
    # 分布式加载数据
    data = dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(data)# num_replicas 和 rank 默认从环境变量中获取
    dataloader=torch.utils.data.DataLoader(data,pin_memory=True,shuffle=False,sampler=sampler,num_workers=args.num_workers,batch_size=args.batch_size) # pin_memory=True 锁业内存 -> 固定内存

    net=model()
    net.to(device)
    #BN->SyncBN
    net=torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) #将 model 中的 BN 替换成分布式的 BN

    if torch.cuda.device_count() > 1:
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank)
    loss_fn=nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    for epoch in range(args.num_epochs):
        # 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        dataloader.sampler.set_epoch(epoch)
        for data, label in dataloader:
            # print(net.module.l1.weight[0][0])
            prediction = net(data)
            loss = loss_fn(prediction.squeeze(1), label.to(torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()     # 所有进程的梯度同步自动进行
            print(loss)

    torch.distributed.destroy_process_group() # 释放资源

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    # parser.add_argument('--local-rank',type=int,default=-1)
    args=parser.parse_args()
    args.batch_size=64   #一次 4*64 = 256个 batches
    args.num_workers=0
    args.num_epochs=100
    # os['NCCL_IB_DISABLE']='1'
    train(args)
