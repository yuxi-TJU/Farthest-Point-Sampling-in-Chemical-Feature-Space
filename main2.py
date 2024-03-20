from load_dataset import *
from sampler import *
from scaler import MaxMinScaler, SignMaxMinScaler
from models import ANN

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import OneCycleLR

from torchmetrics import MeanAbsoluteError, R2Score

import wandb
from results import Stats, normal_metric_point, normal_metric_stream, wrap_metric_stream, wrap_metric_point

    
def nn_task(ratio, property_name='cv', fff=0):
    wandb.init(project="Thermal_ANN", name=f"{property_name}_ANN_{ratio:.1f}")
    stats = Stats(identity=f"{property_name}_{ratio:.1f}_{fff}")

    # X, y, F = load_dataset_batch(*db_args) # 'train_data_critical_property.xlsx', 'A', 2
    X, y = load_dataset_thermal_db("train_data_critical_property.xlsx", 'A', property_name)
    # y = y.squeeze()
    y = y.reshape(-1, 1)

    x_scaler = SignMaxMinScaler(X)
    y_scaler = SignMaxMinScaler(y)

    X_s = x_scaler.transform(X)
    y_s = y_scaler.transform(y)

    X_t = torch.tensor(X_s, dtype=torch.float32).to('cuda')
    y_t = torch.tensor(y_s, dtype=torch.float32).to('cuda')
    dataset = TensorDataset(X_t, y_t)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    trn_idx, val_idx = idx[:int(X.shape[0]*0.8)], idx[int(X.shape[0]*0.8):]

    trn_loader = DataLoader(Subset(dataset, trn_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

    model = ANN(
        input_size=X.shape[1],
        hidden_size=32,
        num_hidden_layers=1,
        output_size=1,
        activation='gelu'
    )
    model.to('cuda')
    MAE = MeanAbsoluteError().to('cuda')
    R2 = R2Score().to('cuda')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, final_div_factor=1e4, total_steps=2000)
    lr_list = np.logspace(-7, -1, 1000)

    for epoch in range(2000):
        model.train()
        train_loss = 0

        # if epoch < len(lr_list):
        #     lr = lr_list[epoch]
        #     for params in optimizer.param_groups:
        #         params['lr'] = lr

        for i, (_in, _t) in enumerate(trn_loader):
            optimizer.zero_grad()
            pred = model(_in)
            loss = criterion(pred, _t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            MAE.update(pred, _t)
            R2.update(pred, _t)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        trn_mae = MAE.compute()
        trn_r2 = R2.compute()
        MAE.reset()
        R2.reset()
        train_loss /= len(trn_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (_in, _t) in enumerate(val_loader):
                pred = model(_in)
                val_loss += criterion(pred, _t)
                MAE.update(pred, _t)
                R2.update(pred, _t)
            val_mae = MAE.compute()
            val_r2 = R2.compute()
            MAE.reset()
            R2.reset()
            val_loss /= len(val_loader)

        wandb.log(
            {
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "lr": lr, 
                "train_mae": trn_mae,
                "val_mae": val_mae,
                "train_r2": trn_r2,
                "val_r2": val_r2
            }, 
            step=epoch+1
        )
        wrap_metric_stream(stats, {
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "lr": lr, 
            "train_mae": trn_mae,
            "val_mae": val_mae,
            "train_r2": trn_r2,
            "val_r2": val_r2
        }, epoch+1)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss} - Val Loss: {val_loss} - LR: {lr}')
    model.eval()
    _p = model(X_t[val_idx, :]).detach().cpu().numpy()
    _p = _p.reshape(-1, 1)
    _p = y_scaler.transform_rev(_p)
    _y = y[val_idx]
    mse = mean_squared_error(_y, _p)
    mae = mean_absolute_error(_y, _p)
    r2 = r2_score(_y, _p)
    wandb.log({
        "final_mse": mse,
        "final_mae": mae,
        "final_r2": r2
    })
    wrap_metric_point(stats, {
        "final_mse": mse,
        "final_mae": mae,
        "final_r2": r2
    }).save(path=f"{property_name}_{ratio:.1f}_{fff}.pkl")
    wandb.finish()

if __name__=='__main__':
    for i in range(1, 11):
        for j in range(3):
            nn_task(i/10, "cv", j)
    
    for i in range(1, 11):
        for j in range(3):
            nn_task(i/10, "cp", j)

    for i in range(1, 11):
        for j in range(3):
            nn_task(i/10, "ct", j)





