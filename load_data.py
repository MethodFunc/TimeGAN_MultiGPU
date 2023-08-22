import pandas as pd
import numpy as np


def dataloader(config, preprocessing=True):
    print("Load Data...")
    raw_data = pd.read_csv(config.path, index_col=config.date_col, parse_dates=True)
    modify_data = raw_data.copy()

    if preprocessing:
        print("Continuous preprocessing...")
        # todo: set your preprocessing data

    modify_data.dropna(inplace=True)

    return modify_data