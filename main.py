from load_dataset import *
from sampler import *
from scaler import SignMaxMinScaler

from config import MODEL_SETTING

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.svm import SVR

import wandb

def metrics_package(y, p):
    mse = mean_squared_error(y, p)
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    return mse, mae, r2

class MyLogger:
    def __init__(self, ratio):
        self.results = {}
        self.results["ratio"] = ratio
        self.stats = {}

    def log(self, results_dict):
        for k, v in results_dict.items():
            if k == "ratio":
                continue
            self.results.setdefault(k, [])
            self.results[k].append(v)

    def stat(self):
        for k, v in self.results.items():
            if k == "ratio":
                continue
            self.stats[k + ".mean"] = np.mean(v) # np.std(v)
        stats = self.stats
        self.results = {}
        self.stats = {}
        return stats

def cv_task(ratio = 0.8):
    wandb.init(project="FPS_Benchmark", name=f"SVM_RS_{ratio:.1f}")
    logger = MyLogger(ratio)

    X, y = load_dataset(path="./dataset.xlsx")
    y = y.reshape(-1, 1)

    x_scaler = SignMaxMinScaler(X)
    y_scaler = SignMaxMinScaler(y)
    
    X_s = x_scaler.transform(X)
    y_s = y_scaler.transform(y)

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

    model: SVR = MODEL_SETTING["SVM"]["model_handler"](
        **MODEL_SETTING["SVM"]["model_params"]
    )
    step = 0
    for k, refer_dict in enumerate(smp.sampling_results[0]["crossval"]):
        for trn_idx, val_idx in zip(refer_dict['train_idx'], refer_dict['val_idx']):
            model.fit(X_s[trn_idx], y_s[trn_idx].squeeze())
            p = model.predict(X_s)
            p = p.reshape(-1, 1)

            p_trn = y_scaler.transform_rev(p[trn_idx])
            if val_idx.size > 0:
                p_val = y_scaler.transform_rev(p[val_idx])
            p_tst = y_scaler.transform_rev(p[refer_dict['test_idx']])

            y_trn = y[trn_idx]
            if val_idx.size > 0:
                y_val = y[val_idx]
            y_tst = y[refer_dict['test_idx']]

            mse_trn, mae_trn, r2_trn = metrics_package(y_trn, p_trn)
            if val_idx.size > 0:
                mse_val, mae_val, r2_val = metrics_package(y_val, p_val)
            else:
                mse_val, mae_val, r2_val = np.nan, np.nan, np.nan
            mse_tst, mae_tst, r2_tst = metrics_package(y_tst, p_tst)

            results = {
                "ratio": ratio,
                "train_loss": mse_trn,
                "val_loss": mse_val,
                "test_loss": mse_tst,
                "train_mae": mae_trn,
                "val_mae": mae_val,
                "test_mae": mae_tst,
                "train_r2": r2_trn,
                "val_r2": r2_val,
                "test_r2": r2_tst
            }
            wandb.log(
                results, step = step
            )
            logger.log(results)
            step += 1
    stats = logger.stat()
    wandb.log(stats)
    wandb.finish()
        
    

if __name__ == "__main__":
    for i in range(1, 11):
        cv_task(i/10)