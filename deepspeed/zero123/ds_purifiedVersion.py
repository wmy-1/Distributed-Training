import torch
import torch.nn as nn
from torch.nn import Sequential,ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader, TensorDataset
import deepspeed

# 纯净版本 仅仅定义必须的内容
class MLP(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=16384, output_dim=10): # 1.3B
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.m = ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(2)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, y): # 模型定义时直接返回损失
        x = F.relu(self.fc1(x))
        for i in range(len(self.m)):
            x = F.relu(self.m[i](x))
        x = self.fc2(x)
        loss = self.criterion(x, y)
        print(x.shape)
        return loss
    
def get_dummy_dataloader(batch_size=2, input_dim=1000, num_samples=100000):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练函数
def train(model, dataloader, epochs=100, device="cuda:0"):
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 使用方式 1.传args,设置args.deepspeed_config 为 config.json 2.直接传config,config = path or dict 3.传config_params,目前与config一致
    model_engine, _, _, _ = deepspeed.initialize(model=model,config={"train_batch_size": 8}) #必须提供batch_size
    # 管理分布式训练环境 torch.distributed.init_process_group() 修改为 deepspeed.init_distributed() 不设置则 DeepSpeed 会在其 initialize 期间自动初始化分布式环境
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(model_engine.local_rank)
            batch_y = batch_y.to(model_engine.local_rank)
            print(batch_x.shape)
            loss = model_engine(batch_x,batch_y)

            print(torch.cuda.memory_allocated() / 1024**2, "MB")
            
            loss.backward()
            optimizer.step()
            # model_engine.backward(loss) # 必须提供优化器
            # model_engine.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    model = MLP()
    dataloader = get_dummy_dataloader()
    train(model, dataloader)



