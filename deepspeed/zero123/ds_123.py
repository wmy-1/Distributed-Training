import torch
import torch.nn as nn
from torch.nn import Sequential,ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
import argparse
import os

# 配置文件 config.json

class MLP(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=4096, output_dim=10): # 48 - 0.75B  96 - 1.5B   
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.m = ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(96)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, y): # 模型定义时直接返回损失
        x = F.relu(self.fc1(x))
        for i in range(len(self.m)):
            x = F.relu(self.m[i](x))
        x = self.fc2(x)
        loss = self.criterion(x, y)
        # print(x.shape)
        return loss
    
def get_dummy_dataset(input_dim=1000, num_samples=100000):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return dataset

# 训练函数
def train(model, dataset, epochs=100, device="cuda:0"):

    """    
    inputs:
    deepspeed.initialize(args=None,model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
               distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config=None,
               mesh_param=None,
               config_params=None)
    outputs:
    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    """
    # 使用方式 1.传args,设置args.deepspeed_config 为 config.json 2.直接传config,config = path or dict 3.传config_params,目前与config一致
    model_engine, _, dataloader, _ = deepspeed.initialize(model=model, \
                                                 training_data=dataset, \
                                                 config="config.json" )
    # 管理分布式训练环境 torch.distributed.init_process_group() 修改为 deepspeed.init_distributed() 不设置则 DeepSpeed 会在其 initialize 期间自动初始化分布式环境
    # print(dir(model_engine))
    for epoch in range(epochs):
        total_loss = 0
        for batch_x,batch_y in dataloader:
            batch_x = batch_x.to(model_engine.local_rank)
            batch_y = batch_y.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                batch_x = batch_x.half()
            if model_engine.bfloat16_enabled():
                batch_x = batch_x.bfloat16()
            print(batch_x.shape)
            loss = model_engine(batch_x,batch_y)

            print(torch.cuda.memory_allocated() / 1024**2, "MB")
            
            model_engine.backward(loss) # 必须提供优化器
            model_engine.step()

            total_loss += loss.item()
            print(f"Batch Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    model = MLP()
    dataset = get_dummy_dataset()
    train(model, dataset)



