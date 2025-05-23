from email_notification import EarlyStopping
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import json
import os
import random
from time import time
from argparse import ArgumentParser
from sklearn.metrics import r2_score

from double_towers import TransGNNModel
from preprocess import DataEncoder, concordance_index1
from util_function import (load_DAVIS, data_process, load_KIBA, load_ChEMBL_kd, load_ChEMBL_pkd,
                           load_BindingDB_kd, load_davis_dataset, load_kiba_dataset)
import sys
import io

# 将标准输出的编码设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

torch.manual_seed(2)
np.random.seed(3)

# Whether to use GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Set loss function
reg_loss_fn = nn.MSELoss()

# Initialize SummaryWriter
log_writer = SummaryWriter(log_dir="./log")


def get_reg_db(db_name):
    """
    Get benchmark dataset for regression
    """
    if db_name.lower() == 'benchmark_davis':
        dataset = load_davis_dataset()
    elif db_name.lower() == 'benchmark_kiba':
        dataset = load_kiba_dataset()
    else:
        raise ValueError('%s not supported' % db_name)
    return dataset


def get_raw_db(db_name):
    """
    Get raw dataset for regression
    """
    # Load Davis data
    if db_name.lower() == 'raw_davis':
        X_drugs, X_targets, y = load_DAVIS(convert_to_log=True)
    # Load Kiba data
    elif db_name.lower() == 'raw_kiba':
        X_drugs, X_targets, y = load_KIBA()
    # Load ChEMBL Kd data
    elif db_name.lower() == 'raw_chembl_kd':
        X_drugs, X_targets, y = load_ChEMBL_kd()
    # Load ChEMBL pKd data
    elif db_name.lower() == 'raw_chembl_pkd':
        X_drugs, X_targets, y = load_ChEMBL_pkd()
    # Load BindingDB Kd data
    elif db_name.lower() == 'raw_bindingdb_kd':
        X_drugs, X_targets, y = load_BindingDB_kd()
    else:
        raise ValueError('%s not supported! ' % db_name)
    
    # 创建包含字典的数据列表
    data_lst = []
    for smiles, protein, aff in zip(X_drugs, X_targets, y):
        data = {}
        data['smiles'] = smiles
        data['protein'] = protein
        data['aff'] = aff
        data_lst.append(data)

    # 假设我们不进行划分，直接将所有数据作为训练集和测试集（根据实际需求调整）
    train_test_dataset = [data_lst, data_lst]

    return train_test_dataset
    # return X_drugs, X_targets, y


def reg_test(data_generator, model):
    """
    Test for regression task
    """
    total_mse = 0
    total_r2 = 0
    total_ci = 0
    total_loss = 0
    num_batches = 0
    all_y_pred = []
    all_y_label = []

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(data_generator):
            d_out, mask_d_out, t_out, mask_t_out, label = data
            d_out, mask_d_out, t_out, mask_t_out, label = d_out.to(device), mask_d_out.to(device), t_out.to(device), mask_t_out.to(device), label.to(device)
            temp = model(d_out.long(), t_out.long(), mask_d_out.long(), mask_t_out.long())

            label = label.float()
            predicts = torch.squeeze(temp, axis=1)

            loss = reg_loss_fn(predicts, label)
            predict_id = torch.squeeze(temp).cpu().numpy()
            label_id = label.cpu().numpy()

            all_y_label.extend(label_id.flatten().tolist())
            all_y_pred.extend(predict_id.flatten().tolist())

            total_loss += loss.item()
            num_batches += 1

    total_label = np.array(all_y_label)
    total_pred = np.array(all_y_pred)

    mse = ((total_label - total_pred) ** 2).mean(axis=0)
    r2 = r2_score(total_label, total_pred)
    ci = concordance_index1(total_label, total_pred)
    avg_loss = total_loss / num_batches

    return (mse, r2, ci, avg_loss)

def save_checkpoint(model, optimizer, scheduler, epoch, log_step, file_path="checkpoint.pth"):
    """
    Save training checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'log_step': log_step
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")


def load_checkpoint(model, optimizer, scheduler, file_path="checkpoint.pth"):
    """
    Load training checkpoint.
    """
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        log_step = checkpoint['log_step']
        print(f"Checkpoint loaded from {file_path}, resuming from epoch {epoch}")
        return epoch, log_step
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0

def main(args):
    """
    Main function
    """
    # Basic setting
    optimal_mse = 10000
    log_iter = 50
    log_step = 0
    optimal_CI = 0

    # print(f"GPU memory before Load model config: {torch.cuda.memory_allocated()} bytes")
    # Load model config
    model_config = json.load(open(args.model_config, 'r'))
    # print(f"GPU memory before Load TransGNNModel: {torch.cuda.memory_allocated()} bytes")
    model = TransGNNModel(model_config)
    print(f"GPU memory before move GPU: {torch.cuda.memory_allocated()} bytes")
    model = model.to(device)
    print(f"GPU memory before Load init Optimizer: {torch.cuda.memory_allocated()} bytes")

    # Optimizer
    optim = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.5, patience=10, verbose=True)

    # Mixed precision training
    scaler = GradScaler()

    # Data Preparation
    # Regression Task - Benchmark Dataset
    print(f"GPU memory before 获取数据: {torch.cuda.memory_allocated()} bytes")
    try:
        trainset, testset = get_raw_db(args.dataset)
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        print("Attempting to load benchmark dataset...")
        trainset, testset = get_reg_db(args.dataset)
    
    trainset_smiles = [d['smiles'] for d in trainset]
    trainset_protein = [d['protein'] for d in trainset]
    trainset_aff = [d['aff'] for d in trainset]

    testset_smiles = [d['smiles'] for d in testset]
    testset_protein = [d['protein'] for d in testset]
    testset_aff = [d['aff'] for d in testset]

    print(f"GPU memory before 压缩重命名: {torch.cuda.memory_allocated()} bytes")

    df_data_t = pd.DataFrame(zip(trainset_smiles, trainset_protein, trainset_aff))
    df_data_t.rename(columns={0: 'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
    df_data_tt = pd.DataFrame(zip(testset_smiles, testset_protein, testset_aff))
    df_data_tt.rename(columns={0: 'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)

    # print(f"GPU memory before DataEncoder: {torch.cuda.memory_allocated()} bytes")

    reg_training_data = DataEncoder(df_data_t.index.values, df_data_t.Label.values, df_data_t)
    reg_train_loader = DataLoader(reg_training_data, batch_size=args.batchsize, shuffle=True, drop_last=False, num_workers=args.workers)
    reg_validation_data = DataEncoder(df_data_tt.index.values, df_data_tt.Label.values, df_data_tt)
    reg_validation_loader = DataLoader(reg_validation_data, batch_size=args.batchsize, shuffle=False, drop_last=False, num_workers=args.workers)

    # Load checkpoint if exists
    start_epoch, log_step = load_checkpoint(model, optim, scheduler)

    # Initial Testing
    print("=====Go for Initial Testing=====")
    with torch.no_grad():
        mse, r2, CI, reg_loss = reg_test(reg_validation_loader, model)
        print("Testing result: MSE: {}, R²: {}, CI: {}".format(mse, r2, CI))
        log_writer.add_scalar("initial/mse", mse, 0)
        log_writer.add_scalar("initial/r2", r2, 0)
        log_writer.add_scalar("initial/CI", CI, 0)
        log_writer.add_scalar("initial/loss", reg_loss, 0)

    # 早停类初始化
    early_stopping = EarlyStopping(patience=20)

    accumulation_steps = 2  # 梯度累积步数
    try:
        # Training
        for epoch in range(args.epochs):
            print("=====Go for Training=====")
            model.train()
            for batch_id, data in enumerate(reg_train_loader):
                d_out, mask_d_out, t_out, mask_t_out, label = data
                d_out, mask_d_out, t_out, mask_t_out, label = d_out.to(device), mask_d_out.to(device), t_out.to(device), mask_t_out.to(device), label.to(device)

                optim.zero_grad()
                with autocast():
                    temp = model(d_out.long(), t_out.long(), mask_d_out.long(), mask_t_out.long())
                    label = label.float()
                    predicts = torch.squeeze(temp)
                    loss = reg_loss_fn(predicts, label)
                    loss = loss / accumulation_steps  # 平均损失

                scaler.scale(loss).backward()

                if (batch_id + 1) % accumulation_steps == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

                if batch_id % log_iter == 0:
                    print("Training at epoch: {}, step: {}, loss is: {}".format(epoch, batch_id, loss.cpu().detach().numpy() * accumulation_steps))
                    log_writer.add_scalar("train/loss", loss.cpu().detach().numpy() * accumulation_steps, log_step)
                    log_step += 1

                # 释放中间变量
                del temp, predicts
                torch.cuda.empty_cache()

            # Validation
            print("=====Go for Validation=====")
            with torch.no_grad():
                mse, r2, CI, reg_loss = reg_test(reg_validation_loader, model)
                print("Validation at epoch: {}, MSE: {}, R²: {}, CI: {}, loss is: {}".format(epoch, mse, r2, CI, reg_loss))
                log_writer.add_scalar("dev/loss", reg_loss, log_step)
                log_writer.add_scalar("dev/mse", mse, log_step)
                log_writer.add_scalar("dev/r2", r2, log_step)
                log_writer.add_scalar("dev/CI", CI, log_step)

                # Save best model
                if mse < optimal_mse:
                    optimal_mse = mse
                    print("Saving the best_model with best MSE...")
                    print("Best MSE: {}".format(optimal_mse))
                    torch.save(model.state_dict(), 'DAVIS_bestMSE_model_reg1.pth')
                if CI > optimal_CI:
                    optimal_CI = CI
                    print("Saving the best_model with best CI...")
                    print("Best CI: {}".format(optimal_CI))
                    torch.save(model.state_dict(), 'DAVIS_bestCI_model_reg1.pth')

                # 检查是否早停
                if early_stopping(reg_loss, mse, CI, epoch, batch_id):
                    print("Early stopping triggered!")
                    break

            # Save checkpoint
            save_checkpoint(model, optim, scheduler, epoch + 1, log_step)

            # Adjust learning rate
            scheduler.step(reg_loss)
    except Exception as e:
        early_stopping.send_training_failure_email(str(e))

    # Print final result
    print("Best MSE: {}".format(optimal_mse))
    print("Best CI: {}".format(optimal_CI))
    torch.save(model.state_dict(), 'DAVIS_final_model_reg1.pth')

    log_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='Start Training...')
    parser.add_argument('-b', '--batchsize', default=32, type=int, metavar='N', help='Batch size')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='Number of workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--dataset', choices=['raw_chembl_pkd', 'raw_chembl_kd', 'raw_bindingdb_kd', 'raw_davis',
                                              'raw_kiba', 'benchmark_davis', 'benchmark_kiba'], default='raw_davis', type=str,
                        metavar='DATASET', help='Select specific dataset for your task')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./config.json', type=str)
    args = parser.parse_args()

    # print(f"GPU memory before main: {torch.cuda.memory_allocated()} bytes")
    beginT = time()
    print("Starting Time: {}".format(beginT))
    main(args)
    endT = time()
    print("Ending Time: {}".format(endT))
    print("Duration is: {}".format(endT - beginT))