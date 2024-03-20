from load_dataset import *
from sampler import *
from scaler import MaxMinScaler, SignMaxMinScaler
from models import model_initiator, ANN

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

import wandb
import optuna
    
def train_fn(db_args, svm_config):
    wandb.init(project="SupportVectorMachine", name="SVM_SMO_0.2")

    # X, y, F = load_dataset_batch(*db_args) # 'train_data_critical_property.xlsx', 'A', 2
    X, y = load_dataset()
    # y = y.squeeze()
    y = y.reshape(-1, 1)

    x_scaler = SignMaxMinScaler(X)
    y_scaler = SignMaxMinScaler(y)

    X_s = x_scaler.transform(X)
    y_s = y_scaler.transform(y)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    trn_idx, val_idx = idx[:int(X.shape[0]*0.2)], idx[int(X.shape[0]*0.2):]

    wandb.config.update(svm_config)

    model = SVR(
        **svm_config
    )
    model.fit(X_s[trn_idx], y_s[trn_idx].squeeze())

    y_trn_pred = model.predict(X_s[trn_idx])
    y_pred = model.predict(X_s[val_idx])
    
    train_loss = mean_squared_error(y_s[trn_idx], y_trn_pred)
    val_loss = mean_squared_error(y_s[val_idx], y_pred)
    train_mae = mean_absolute_error(y_s[trn_idx], y_trn_pred)
    val_mae = mean_absolute_error(y_s[val_idx], y_pred)
    train_r2 = r2_score(y_s[trn_idx], y_trn_pred)
    val_r2 = r2_score(y_s[val_idx], y_pred)

    wandb.log(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
    )
    print(f'Train Loss: {train_loss} - Val Loss: {val_loss}')

    wandb.finish()
    return val_loss

def objective(trial):
    svm_config = {
        "kernel": "rbf",
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "epsilon": trial.suggest_loguniform("epsilon", 1e-2, 1e2),
        "tol": trial.suggest_loguniform("tol", 1e-6, 1e-2),
        "gamma": trial.suggest_loguniform("gamma", 1e-2, 1e2)
    }

    val_loss = train_fn([], svm_config)
    return val_loss

if __name__=='__main__':

    # study = optuna.create_study(
    #     direction="minimize",
    #     sampler=optuna.samplers.TPESampler(),
    # )
    # study.optimize(objective, n_trials=200)

    for i in range(10):
        svm_config = {
            "kernel": "rbf",
            "C": 36,
            "epsilon": 0.012,
            "tol": 1.5e-6, 
            "gamma": 1.2
        }
        train_fn([], svm_config)





