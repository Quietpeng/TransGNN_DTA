# TransGNN_DTA 项目文档
 [English](README.md) | [中文](README_zh.md) 


## 介绍
TransGNN_DTA 是一个用于药物 - 靶点亲和力预测的项目，实现了结合 Transformer 和 GNN 技术的 TransGNN 模型。该模型旨在更准确地预测药物与靶点之间的结合亲和力，支持 DAVIS 和 KIBA 等多个基准数据集，并提供了相应的训练和测试脚本。同时，项目还包含了数据预处理、模型训练、早停机制以及邮件通知等功能，方便用户进行实验和结果监控。

相关论文正在审核中



## 模型方法
### 模型架构
TransGNN 模型主要由以下几个核心组件构成：
1. **增强嵌入层（Enhanced Embedding）**：对药物和靶点序列进行编码，包含词嵌入（word embeddings）和位置嵌入（position embeddings），并添加了归一化（normalization）和丢弃（dropout）操作，以提高模型的泛化能力。
2. **编码器模块（Encoder Module）**：通过堆叠多个编码器层，每个层包含注意力机制（attention mechanism）和前馈神经网络（feed - forward neural network），捕捉序列中的上下文信息。
3. **GNN 层（GNN Layer）**：对药物和靶点序列进行消息传递（message passing），以捕捉其图结构信息，增强模型对分子结构和相互作用的理解。
4. **解码器模块（Decoder Module）**：将处理后的特征映射到一维输出，用于预测药物与靶点之间的结合亲和力。

### 训练方法
- **损失函数**：使用均方误差损失（MSE Loss）作为回归任务的损失函数，用于衡量模型预测值与真实值之间的差异。
- **优化器**：采用 AdamW 优化器进行参数更新，结合学习率调度器（ReduceLROnPlateau）根据验证集损失动态调整学习率。
- **混合精度训练**：使用 PyTorch 的 `GradScaler` 和 `autocast` 实现混合精度训练，加速训练过程并减少内存占用。



## 快速开始
> [!IMPORTANT]
>详细使用指导请查看[示例笔记本](Example.ipynb) 

### 1. 训练
#### 1.1 环境配置
根据实际情况自行配置 torch 与 CUDA 环境。实验采用的配置如下：
- 框架名称：PyTorch
- 框架版本：2.3.0
- Python 版本：3.12 (ubuntu22.04)
- CUDA 版本：12.4
- GPU：vGPU - 32GB(32GB)*1

可以使用以下命令创建并激活虚拟环境：
```bash
python -m venv transgnn_env
source transgnn_env/bin/activate
```

#### 1.2 克隆仓库
在空文件夹下克隆 GitHub 仓库：
```bash
git clone https://github.com/Quietpeng/TransGNN_DTA.git
cd TransGNN_DTA
```

#### 1.3 安装依赖
根据 `requirements.txt` 文件使用以下命令安装所需的包：
```bash
pip install -r requirements.txt
```
如果你需要后台管理，可以安装 `screen`：
```bash
sudo apt-get install screen
```

#### 1.4 开启 tensorboard 转发端口（可选）
若需要监控训练过程，可以开启 tensorboard 转发端口。首先，启动 tensorboard：
```bash
tensorboard --logdir=./log
```
然后，转发端口（假设本地端口为 6006，远程服务器端口为 6006）：
```bash
ssh -L 6006:localhost:6006 user@server_ip
```
在浏览器中访问 `http://localhost:6006` 即可查看训练日志。

#### 1.5 超参数与命令行参数解释与选择
- **超参数**：主要在 `config.json` 文件中配置，包括 `drug_max_seq`（药物序列最大长度）、`target_max_seq`（靶点序列最大长度）、`emb_size`（嵌入层维度）等。
- **命令行参数**：可以在运行 `train_reg.py` 脚本时指定，常用参数如下：
  - `--model_config`：指定模型配置文件路径，默认为 `config.json`。
  - `--lr`：学习率，默认为 0.001。
  - `--b`：批量大小，默认为 32。
  - `--epochs`：训练轮数，默认为 100。
  - `--dataset`：指定数据集名称，支持 `benchmark_davis`、`benchmark_kiba`、`raw_davis`、`raw_kiba` 等。

例如，使用以下命令开始训练：
```bash
python train_reg.py --model_config config.json --lr 0.001 --b 32 --epochs 100 --dataset raw_davis
```

### 2. 使用
训练完成后，你可以使用训练好的模型进行预测。可以修改 `train_reg.py` 脚本或编写新的脚本，加载训练好的模型并对新数据进行预测。以下是一个简单的示例：
```python
import torch
import json
from preprocess import drug_encoder, target_encoder
from double_towers import TransGNNModel

# 示例数据（药物SMILES与蛋白质序列）
DRUG_EXAMPLE = "CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)Cl"
PROTEIN_EXAMPLE = "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSPGEPSPPSIHPGKSDLIVRVGDEIRLLCTDP"

# 配置文件与模型路径
MODEL_CONFIG_PATH = "config.json"
MODEL_PATH = "./models/DAVIS_bestCI_model_reg1.pth"  # 替换为实际训练好的模型路径

# 加载模型配置
model_config = json.load(open(MODEL_CONFIG_PATH, 'r'))
# 初始化模型
model = TransGNNModel(model_config)
# 检查是否有可用的 GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
model = model.to(device)

# 加载训练好的模型权重
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    # 调整 decoder 层的输入维度
    model.decoder[0] = nn.Linear(list(checkpoint['decoder.0.weight'].shape)[1], model.decoder[0].out_features).to(device)
    model.load_state_dict(checkpoint)
    print("Successfully loaded model state from checkpoint.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

model.eval()

# 序列编码（返回特征向量与掩码）
d_out, mask_d_out = drug_encoder(DRUG_EXAMPLE)
t_out, mask_t_out = target_encoder(PROTEIN_EXAMPLE)

# 转换为张量并移动至设备
d_tensor = torch.LongTensor(d_out).unsqueeze(0).to(device)
mask_d_tensor = torch.LongTensor(mask_d_out).unsqueeze(0).to(device)
t_tensor = torch.LongTensor(t_out).unsqueeze(0).to(device)
mask_t_tensor = torch.LongTensor(mask_t_out).unsqueeze(0).to(device)

# 执行预测
with torch.no_grad():
    prediction = model(d_tensor, t_tensor, mask_d_tensor, mask_t_tensor).cpu().numpy()

print(f"亲和力预测值：{prediction[0][0]:.4f}")  # 输出格式化为4位小数
```



## 说明

### 数据准备
确保数据集放置在正确的目录下。代码支持 DAVIS 和 KIBA 等多个数据集，不同的数据集需要根据代码逻辑放置在相应的目录中。例如，基准数据集的路径为 `dataset/regression/benchmark`，原始数据集的路径为 `dataset/regression`。

### 模型检查点
代码提供了保存和加载检查点的功能。在训练过程中，使用 `save_checkpoint` 函数保存训练状态，使用 `load_checkpoint` 函数加载训练状态，以便在中断训练后继续训练。例如：
```python
# 保存检查点
save_checkpoint(model, optimizer, scheduler, epoch, log_step, file_path="checkpoint.pth")

# 加载检查点
start_epoch, log_step = load_checkpoint(model, optimizer, scheduler, file_path="checkpoint.pth")
```

### 早停机制
代码使用 `EarlyStopping` 类进行早停。当验证集损失在连续多个轮次内没有改善时，停止训练并发送邮件通知（如果配置了邮件功能）。可以根据需要调整 `patience` 参数（早停的耐心值）。例如：
```python
early_stopping = EarlyStopping(patience=20)
for epoch in range(args.epochs):
    # 训练代码
    mse, r2, CI, reg_loss = reg_test(reg_validation_loader, model)
    if early_stopping(reg_loss, mse, CI, epoch, batch_id):
        break
```

### 邮件通知
项目支持训练早停和训练失败时的邮件通知功能。需要在 `config.json` 文件中配置邮箱信息，示例如下：
```json
{
    "email": {
        "enabled": true,
        "sender_email": "your_email@example.com",
        "sender_password": "your_email_password",
        "receiver_email": "recipient_email@example.com",
        "smtp_server": "smtp.example.com",
        "smtp_port": 465
    }
}
```



## 致谢

感谢本项目的所有贡献者。他们的努力和贡献使得本项目得以实现。同时，感谢开源社区提供的各种工具和库，为项目的开发和实现提供了便利。