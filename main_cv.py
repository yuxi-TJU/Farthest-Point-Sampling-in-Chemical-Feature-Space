from load_dataset import *
from sampler import *
from scaler import SignMaxMinScaler

from config import MODEL_SETTING, ANN

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.svm import SVR

import wandb
from results import wrap_metric_point, Stats

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import OneCycleLR

from torchmetrics import MeanAbsoluteError, R2Score, MeanSquaredError


def metrics_package(y, p):
    mse = mean_squared_error(y, p)
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    return mse, mae, r2



def cv_task(ratio = 0.8, property_name = 'cv'):
    print(f"task {property_name} {ratio:.1f}")

    # X, y = load_dataset(path="./dataset.xlsx")
    X, y = load_dataset_thermal_db("train_data_critical_property.xlsx", 'A', property_name)
    y = y.reshape(-1, 1)

    x_scaler = SignMaxMinScaler(X)
    y_scaler = SignMaxMinScaler(y)
    
    X_s = x_scaler.transform(X)
    y_s = y_scaler.transform(y)

    X_t = torch.tensor(X_s, dtype=torch.float32).to('cuda')
    y_t = torch.tensor(y_s, dtype=torch.float32).to('cuda')
    dataset = TensorDataset(X_t, y_t)

    smp = RandomSampler(
        strategy='random',
        seed = 66,
        n = X_s.shape[0], 
        r_test = 0.2, 
        r_train = ratio, 
        num_tries_inner = 5, 
        test_crossval = True,
        # feature = X_s
    )
    smp.test_split()
    smp.sampling_split()

    model: ANN = MODEL_SETTING["ANN"]["model_handler"](
        input_size=X.shape[1],
        **MODEL_SETTING["ANN"]["model_params"]
    )
    model.to('cuda')
    MAE = MeanAbsoluteError().to('cuda')
    MSE = MeanSquaredError().to('cuda')
    R2 = R2Score().to('cuda')

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, final_div_factor=1e4, total_steps=2000)

    step = 0
    stats_list = []
    for k, refer_dict in enumerate(smp.sampling_results[0]["crossval"]):
        for trn_idx, val_idx in zip(refer_dict['train_idx'], refer_dict['val_idx']):
            
            model: ANN = MODEL_SETTING["ANN"]["model_handler"](
                input_size=X.shape[1],
                **MODEL_SETTING["ANN"]["model_params"]
            )
            model.to('cuda')
            MAE = MeanAbsoluteError().to('cuda')
            MSE = MeanSquaredError().to('cuda')
            R2 = R2Score().to('cuda')

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = OneCycleLR(optimizer, max_lr=1e-2, final_div_factor=1e4, total_steps=2000)

            stats = Stats(identity=f"{property_name}_{ratio:.1f}")
            trn_loader = DataLoader(Subset(dataset, trn_idx), batch_size=32, shuffle=True)
            tst_loader = DataLoader(Subset(dataset, refer_dict['test_idx']), batch_size=32, shuffle=False)

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
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']

                train_loss /= len(trn_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (_in, _t) in enumerate(tst_loader):
                    pred = model(_in)
                    val_loss += criterion(pred, _t)
                    MSE.update(pred, _t)
                    MAE.update(pred, _t)
                    R2.update(pred, _t)
                tst_mae = MAE.compute()
                tst_mse = MSE.compute()
                tst_r2 = R2.compute()
                MSE.reset()
                MAE.reset()
                R2.reset()

                for i, (_in, _t) in enumerate(trn_loader):
                    pred = model(_in)
                    MSE.update(pred, _t)
                    MAE.update(pred, _t)
                    R2.update(pred, _t)
                trn_mae = MAE.compute()
                trn_mse = MSE.compute()
                trn_r2 = R2.compute()
                MSE.reset()
                MAE.reset()
                R2.reset()

            metrics = {
                "final_tst_mse": tst_mse.item(),
                "final_tst_mae": tst_mae.item(),
                "final_tst_r2": tst_r2.item(),
                "final_trn_mse": trn_mse.item(),
                "final_trn_mae": trn_mae.item(),
                "final_trn_r2": trn_r2.item()
            }
            wrap_metric_point(stats, metrics)
            stats_list.append(stats)
    import pickle as pkl
    with open(f"rs_{property_name}_{ratio:.1f}_stats.pkl", "wb") as f:
        pkl.dump(stats_list, f)
  

if __name__ == "__main__":
    for i in range(1, 11):
        cv_task(i/10, 'cv')

    for i in range(1, 11):
        cv_task(i/10, 'cp')

    for i in range(1, 11):
        cv_task(i/10, 'ct')