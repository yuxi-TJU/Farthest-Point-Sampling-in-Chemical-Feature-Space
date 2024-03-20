import numpy as np
import pickle as pkl

import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PStat:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.name}: {self.value}"
    
class SStat:
    def __init__(self, name):
        self.name = name
        self.values = []
        self.steps = []

    def __repr__(self):
        return f"{self.name}: {self.values[-1]} at step {self.steps[-1]}"
    
    def update(self, value, step):
        self.values.append(value)
        self.steps.append(step)

class Stats:
    def __init__(self, identity, path=None, load=False):
        if path and load:
            self.load(path)
        else:
            self._stats = {"identity": identity, "points": {}, "streams": {}}

    @property
    def identity(self):
        return self._stats["identity"]

    @property
    def points(self):
        return self._stats["points"]
    
    @property
    def streams(self):
        return self._stats["streams"]
    
    def add_point(self, name, value, overwrite=False):
        if self.points.get(name) and overwrite is False:
            raise ValueError(f"Point {name} already exists")
        self.points[name] = PStat(name, value)

    def add_point_dict(self, point_dict):
        for name, value in point_dict.items():
            self.add_point(name, value)

    def add_stream(self, name):
        if self.streams.get(name):
            raise ValueError(f"Stream {name} already exists")
        self.streams[name] = SStat(name)

    def update_stream(self, name, value, step):
        if self.streams.get(name) is None:
            self.add_stream(name)
        self.streams[name].update(value, step)

    def update_stream_dict(self, stream_dict, step):
        for name, value in stream_dict.items():
            self.update_stream(name, value, step)

    def load(self, path):
        with open(path, 'rb') as f:
            self._stats = pkl.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self._stats, f)


# utils function
def pre_process(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()
    elif isinstance(arr, np.ndarray):
        pass
    elif isinstance(arr, list):
        arr = np.array(arr)
    else:
        raise ValueError(f"Unsupported type {type(arr)}")
    if arr.ndim > 1:
        arr = arr.squeeze()
    return arr

def normal_metric_stream(stats: Stats, y, p, step, prefix=""):
    if prefix:
        prefix += "_"

    y = pre_process(y)
    p = pre_process(p)

    mae = mean_absolute_error(y, p)
    mse = mean_squared_error(y, p)
    r2 = r2_score(y, p)
    stats = {
        f"{prefix}mae": mae,
        f"{prefix}mse": mse,
        f"{prefix}r2": r2
    }

    stats.update_stream_dict(stats, step)
    return stats

def normal_metric_point(stats: Stats, y, p, prefix=""):
    if prefix:
        prefix += "_"

    y = pre_process(y)
    p = pre_process(p)

    mae = mean_absolute_error(y, p)
    mse = mean_squared_error(y, p)
    r2 = r2_score(y, p)
    stats = {
        f"{prefix}mae": mae,
        f"{prefix}mse": mse,
        f"{prefix}r2": r2
    }

    stats.add_point_dict(stats)
    return stats

def wrap_metric_stream(stats: Stats, metric_dict, step):
    for name, value in metric_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        stats.update_stream(name, value, step)
    return stats

def wrap_metric_point(stats: Stats, metric_dict):
    for name, value in metric_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        stats.add_point(name, value)
    return stats