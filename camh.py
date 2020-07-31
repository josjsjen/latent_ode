import torch
import os
import pandas as pd
import numpy as np


class Simulated(object):
    def __init__(self, root,
                 quantization=0.1, n_samples=None, device=torch.device("cpu")):
        self.root = root
        self.reduce = "average"
        self.quantization = quantization
        self.csv = pd.read_csv(self.root)
        self.device = device

    def processing(self):
        df = self.csv
        df = df.drop(['time'], axis=1)

        self.dataset_obj = []

        max_timestamps = 0
        for id in df.id.values:
            if df.loc[df.id == id].iloc[:, 1].shape[0] > max_timestamps:
                max_timestamps = df.loc[df.id == id].iloc[:, 1].shape[0]

        total_unique_ids = df['id'].unique()
        for i in total_unique_ids:
            patient_id = i
            tt = np.arange(max_timestamps)
            vals = df.loc[df.id == i].iloc[:, 2:].values
            if max_timestamps - vals.shape[0] >0:
                dup = np.stack([vals[-1] for _ in range(max_timestamps - vals.shape[0])], axis=0)
                vals = np.vstack((vals, dup))

            masks = np.copy(vals)
            masks[~np.isnan(masks)] = 1
            masks[np.isnan(masks)] = 0

            labels=np.zeros(1)

            # todo: confirm if this is the right way to impute missingness
            vals = np.nan_to_num(vals, nan=0)

            tt = torch.tensor(tt).to("cpu").to(dtype=torch.float32)
            vals = torch.tensor(vals).to("cpu").to(dtype=torch.float32)
            masks = torch.tensor(masks).to("cpu").to(dtype=torch.float32)
            labels = torch.tensor(labels).to("cpu").to(dtype=torch.float32)

            self.dataset_obj.append((patient_id, tt, vals, masks, labels))
        return self.dataset_obj
