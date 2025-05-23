
# TransGNN_DTA Project Documentation  
[English](README.md) | [中文](README_zh.md)  


## Introduction  
TransGNN_DTA is a project for drug-target affinity prediction, implementing the TransGNN model that combines Transformer and GNN technologies. The model aims to more accurately predict the binding affinity between drugs and targets, supports multiple benchmark datasets such as DAVIS and KIBA, and provides corresponding training and testing scripts. Meanwhile, the project includes functions like data preprocessing, model training, early stopping, and email notifications to facilitate users' experiments and result monitoring.  

Related papers are under review.  


## Model Methodology  
### Model Architecture  
The TransGNN model consists of the following core components:  
1. **Enhanced Embedding Layer**: Encodes drug and target sequences, including word embeddings and position embeddings, with normalization and dropout operations to improve the model's generalization ability.  
2. **Encoder Module**: Captures contextual information in sequences by stacking multiple encoder layers, each containing an attention mechanism and a feed-forward neural network.  
3. **GNN Layer**: Performs message passing on drug and target sequences to capture their graph structure information, enhancing the model's understanding of molecular structures and interactions.  
4. **Decoder Module**: Maps processed features to one-dimensional outputs for predicting drug-target binding affinity.  

### Training Approach  
- **Loss Function**: Uses Mean Squared Error (MSE) loss as the loss function for regression tasks to measure the difference between the model's predicted values and true values.  
- **Optimizer**: Employs the AdamW optimizer for parameter updating, combined with a learning rate scheduler (ReduceLROnPlateau) to dynamically adjust the learning rate based on the validation set loss.  
- **Mixed Precision Training**: Implements mixed precision training using PyTorch's `GradScaler` and `autocast` to accelerate training and reduce memory usage.  


## Quick Start  
> [!IMPORTANT]  
> For detailed usage instructions, please refer to the [Example Notebook](Example_en.ipynb).  

### 1. Training  
#### 1.1 Environment Setup  
Configure the PyTorch and CUDA environment according to your actual situation. The experimental configuration is as follows:  
- Framework: PyTorch  
- Framework Version: 2.3.0  
- Python Version: 3.12 (ubuntu22.04)  
- CUDA Version: 12.4  
- GPU: vGPU-32GB(32GB)*1  

You can create and activate a virtual environment using the following commands:  
```bash  
python -m venv transgnn_env  
source transgnn_env/bin/activate  
```  

#### 1.2 Clone the Repository  
Clone the GitHub repository in an empty folder:  
```bash  
git clone https://github.com/Quietpeng/TransGNN_DTA.git  
cd TransGNN_DTA  
```  

#### 1.3 Install Dependencies  
Install the required packages using the following command based on the `requirements.txt` file:  
```bash  
pip install -r requirements.txt  
```  
If you need background management, install `screen`:  
```bash  
sudo apt-get install screen  
```  

#### 1.4 Enable TensorBoard Port Forwarding (Optional)  
To monitor the training process, enable TensorBoard port forwarding. First, start TensorBoard:  
```bash  
tensorboard --logdir=./log  
```  
Then, forward the port (assuming the local port is 6006 and the remote server port is 6006):  
```bash  
ssh -L 6006:localhost:6006 user@server_ip  
```  
View the training logs by visiting `http://localhost:6006` in your browser.  

#### 1.5 Explanation and Selection of Hyperparameters and Command-Line Arguments  
- **Hyperparameters**: Mainly configured in the `config.json` file, including `drug_max_seq` (maximum drug sequence length), `target_max_seq` (maximum target sequence length), `emb_size` (embedding layer dimension), etc.  
- **Command-Line Arguments**: Can be specified when running the `train_reg.py` script. Common arguments include:  
  - `--model_config`: Specify the path to the model configuration file, defaulting to `config.json`.  
  - `--lr`: Learning rate, defaulting to 0.001.  
  - `--b`: Batch size, defaulting to 32.  
  - `--epochs`: Number of training epochs, defaulting to 100.  
  - `--dataset`: Specify the dataset name, supporting `benchmark_davis`, `benchmark_kiba`, `raw_davis`, `raw_kiba`, etc.  

For example, start training using the following command:  
```bash  
python train_reg.py --model_config config.json --lr 0.001 --b 32 --epochs 100 --dataset raw_davis  
```  

### 2. Inference  
After training, you can use the trained model for prediction. You can modify the `train_reg.py` script or write a new script to load the trained model and predict new data. Below is a simple example:  
```python  
import torch  
import json  
from preprocess import drug_encoder, target_encoder  
from double_towers import TransGNNModel  

# Example data (drug SMILES and protein sequence)  
DRUG_EXAMPLE = "CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)Cl"  
PROTEIN_EXAMPLE = "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSPGEPSPPSIHPGKSDLIVRVGDEIRLLCTDP"  

# Paths to configuration file and model  
MODEL_CONFIG_PATH = "config.json"  
MODEL_PATH = "./models/DAVIS_bestCI_model_reg1.pth"  # Replace with the actual path of the trained model  

# Load model configuration  
model_config = json.load(open(MODEL_CONFIG_PATH, 'r'))  
# Initialize the model  
model = TransGNNModel(model_config)  
# Check for available GPU  
use_cuda = torch.cuda.is_available()  
device = torch.device('cuda:0' if use_cuda else 'cpu')  
model = model.to(device)  

# Load trained model weights  
try:  
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)  
    # Adjust the input dimension of the decoder layer  
    model.decoder[0] = nn.Linear(list(checkpoint['decoder.0.weight'].shape)[1], model.decoder[0].out_features).to(device)  
    model.load_state_dict(checkpoint)  
    print("Successfully loaded model state from checkpoint.")  
except Exception as e:  
    print(f"Error loading checkpoint: {e}")  

model.eval()  

# Sequence encoding (returns feature vectors and masks)  
d_out, mask_d_out = drug_encoder(DRUG_EXAMPLE)  
t_out, mask_t_out = target_encoder(PROTEIN_EXAMPLE)  

# Convert to tensors and move to device  
d_tensor = torch.LongTensor(d_out).unsqueeze(0).to(device)  
mask_d_tensor = torch.LongTensor(mask_d_out).unsqueeze(0).to(device)  
t_tensor = torch.LongTensor(t_out).unsqueeze(0).to(device)  
mask_t_tensor = torch.LongTensor(mask_t_out).unsqueeze(0).to(device)  

# Perform prediction  
with torch.no_grad():  
    prediction = model(d_tensor, t_tensor, mask_d_tensor, mask_t_tensor).cpu().numpy()  

print(f"Predicted affinity value: {prediction[0][0]:.4f}")  # Output formatted to 4 decimal places  
```  


## Instructions  
### Data Preparation  
Ensure the dataset is placed in the correct directory. The code supports multiple datasets such as DAVIS and KIBA, which need to be placed in corresponding directories according to the code logic. For example, the path for benchmark datasets is `dataset/regression/benchmark`, and the path for raw datasets is `dataset/regression`.  

### Model Checkpoints  
The code provides functions to save and load checkpoints. During training, use the `save_checkpoint` function to save the training state and `load_checkpoint` to resume training after an interruption. For example:  
```python  
# Save checkpoint  
save_checkpoint(model, optimizer, scheduler, epoch, log_step, file_path="checkpoint.pth")  

# Load checkpoint  
start_epoch, log_step = load_checkpoint(model, optimizer, scheduler, file_path="checkpoint.pth")  
```  

### Early Stopping Mechanism  
The code uses the `EarlyStopping` class for early stopping. When the validation set loss does not improve for consecutive epochs, training stops, and an email notification is sent (if the email function is configured). You can adjust the `patience` parameter (the number of epochs to wait for improvement) as needed. For example:  
```python  
early_stopping = EarlyStopping(patience=20)  
for epoch in range(args.epochs):  
    # Training code  
    mse, r2, CI, reg_loss = reg_test(reg_validation_loader, model)  
    if early_stopping(reg_loss, mse, CI, epoch, batch_id):  
        break  
```  

### Email Notification  
The project supports email notifications for early stopping and training failures. You need to configure email information in the `config.json` file, as shown in the example below:  
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


## Acknowledgments  
Thanks to all contributors to this project. Their efforts and contributions have made this project possible. We also thank the open-source community for providing various tools and libraries that facilitated the development and implementation of this project.