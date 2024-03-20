from sklearn.svm import SVR
from models import ANN

MODEL_SETTING = {
    "SVM": {
        "model_type": "SVR",
        "model_params": {
            "kernel": "rbf",
            "C": 36,
            "epsilon": 0.012,
            "tol": 1.5e-6, 
            "gamma": 1.2
        },
        "model_handler": SVR
    },
    "ANN": {
        "model_type": "ANN_torch",
        "model_params": {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "output_size": 1,
            "activation": 'gelu'
        },
        "model_handler": ANN
    }
}
