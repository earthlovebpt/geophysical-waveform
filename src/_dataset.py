
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        mode = "train",
    ):
        self.cfg = cfg
        self.mode = mode

        self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self, ):

        # Select rows
        df= pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        if self.cfg.subsample is not None:
            df= df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df= df[df["fold"] != 0]
        else:
            df= df[df["fold"] == 0]


        data = []
        labels = []
        records = []
        mmap_mode = "r"

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=self.cfg.local_rank != 0):
            row= row.to_dict()

            # Hacky way to get exact file name
            p1 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"])
            p2 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            p3 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"])
            p4 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            farr= glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)

            # Map to lbl fpath
            farr= farr[0]
            flbl= farr.replace('seis', 'vel').replace('data', 'model')

            # Load
            arr= np.load(farr, mmap_mode=mmap_mode)
            lbl= np.load(flbl, mmap_mode=mmap_mode)

            # Append
            data.append(arr)
            labels.append(lbl)
            records.append(row["dataset"])

        return data, labels, records

    def __getitem__(self, idx):
        row_idx= idx // 500
        col_idx= idx % 500

        d= self.records[row_idx]
        x= self.data[row_idx][col_idx, ...]
        y= self.labels[row_idx][col_idx, ...]

        # Augs
        if self.mode == "train":

            # Temporal flip
            if np.random.random() < 0.5:
                x= x[::-1, :, ::-1]
                y= y[..., ::-1]

        x= x.copy()
        y= y.copy()

        return x, y

    def __len__(self, ):
        return len(self.records) * 500
