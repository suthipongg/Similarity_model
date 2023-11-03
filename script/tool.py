import os
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROOT_NFS = Path("/app/nfs_clientshare")
ROOT_NFS_TEST = ROOT_NFS / "mew/project/Similarity_model"
ROOT_NFS_DATA = ROOT_NFS / "Datasets"

blacklist = [".DS_Store", "บาร์โค้ด", "barcode", "desktop.ini", ".csv"]
def in_blacklist(file):
    for bl in blacklist:
        if bl.lower() in file.lower():
            return True
    return False

def scan_directory(path):
    # Initialize a list to store the paths of JPG files
    df = []

    # Use os.walk to traverse the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if not in_blacklist(file):
                df.append([os.path.join(root, file), file, os.path.basename(root)])
    df = pd.DataFrame(df, columns = ['path_img', 'file_name', 'classes'])
    print(f"amount of all image : {len(df)}")
    return df

def standardize_feature(arr):
    return (arr-arr.mean())/arr.std()

def to_unit_len(vector):
    return vector / np.linalg.norm(vector)