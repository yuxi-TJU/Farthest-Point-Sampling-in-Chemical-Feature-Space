import numpy as np
import pandas as pd

def load_dataset(path='./dataset.xlsx'):
    dataset = pd.read_excel(
        path,
        index_col=None
    )

    y = dataset['property'].values
    X = dataset.iloc[:, 1:].values
    return X, y


def load_dataset_thermal_db(path, sheet_name='A', property_name=0):
    property_map = {
        "cv": 0,
        "ct": 1,
        "cp": 2,
    }
    prop = pd.read_excel(
        path,
        index_col=None,
        sheet_name='Property'
    )
    if prop.shape[1] > 1:
        prop = prop.iloc[:, property_map[property_name]]

    y = prop.values
    desc = pd.read_excel(
        path,
        index_col=None,
        sheet_name=sheet_name
    )
    X = desc.values

    return X, y