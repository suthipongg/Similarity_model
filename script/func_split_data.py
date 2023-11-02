from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

import pandas as pd

from tqdm.notebook import tqdm
from script.tool import ROOT_NFS_DATA

class split_data:
    def __init__(self, data_path='Cosmenet_product_20231018', data_csv='datas_20231018.csv'):
        self.path_dataset = ROOT_NFS_DATA / data_path
        self.df_pd = pd.read_csv(self.path_dataset / data_csv)
    
    def filter_data(self, filter_img=8):
        self.filter_img = filter_img
        self.group_df = self.df_pd.groupby(['labels'])['labels'].count().reset_index(name='count').sort_values(['count'], ascending=False)
        filter_img_2_to_n = self.group_df[(self.group_df['count'] <= filter_img) & (self.group_df['count'] > 1)]['labels'].values
        filter_img_1_to_n = self.group_df[self.group_df['count'] <= filter_img]['labels'].values

        self.df_more_n = self.df_pd[~self.df_pd['labels'].isin(filter_img_1_to_n)]
        self.df_2_to_n = self.df_pd[self.df_pd['labels'].isin(filter_img_2_to_n)]
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        train_2_to_n, test_2_to_n = skf.split(self.df_2_to_n, self.df_2_to_n['labels']).__next__()
        df_2_to_n_train = self.df_2_to_n.iloc[train_2_to_n]
        df_2_to_n_test = self.df_2_to_n.iloc[test_2_to_n]
        return df_2_to_n_train, df_2_to_n_test
    
    def train_test_split(self, test_size=0.2, random_state=42, n_splits=1):
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        for train_index, test_index in sss.split(self.df_more_n, self.df_more_n['labels']):
            train_set = self.df_more_n.iloc[train_index]
            test_set = self.df_more_n.iloc[test_index]

        return train_set, test_set
    
    def validate_split(self, test_size=0.18, random_state=42, n_splits=1):
        df_count_n = self.group_df[self.group_df['count'] > self.filter_img]
        group_df_count_n = df_count_n.groupby(['count'])['count'] \
            .count().reset_index(name='counter_count').sort_values(['counter_count'], ascending=False)
        counter_count_1 = group_df_count_n[group_df_count_n["counter_count"] == 1]["count"].values
        df_count_n.loc[df_count_n["count"].isin(counter_count_1), "count"] = 101
        
        sss_val = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        split_idx, val_idx = sss_val.split(df_count_n, df_count_n['count']).__next__()
        split_class = df_count_n.iloc[split_idx]["labels"].values
        val_class = df_count_n.iloc[val_idx]["labels"].values
        return split_class, val_class
        
    def split_data(self, test_size=0.2, random_state=42, n_splits=1):
        df_2_to_n_train, df_2_to_n_test = self.filter_data(8)
        train_set, test_set = self.train_test_split(test_size=test_size, random_state=random_state, n_splits=n_splits)
        split_class, val_class = self.validate_split()
        self.df_train_split = train_set[train_set["labels"].isin(split_class)].reset_index(drop=True)
        self.df_test_split = test_set[test_set["labels"].isin(split_class)].reset_index(drop=True)
        self.df_train_val = train_set[train_set["labels"].isin(val_class)]
        self.df_test_val = test_set[test_set["labels"].isin(val_class)]
        
        self.df_train_val_mix = pd.concat([self.df_train_val, df_2_to_n_train]).reset_index(drop=True)
        self.df_test_val_mix = pd.concat([self.df_test_val, df_2_to_n_test]).reset_index(drop=True)
    
    def get_train_test(self):
        return self.df_train_split, self.df_test_split
    
    def get_validate(self):
        return self.df_train_val_mix, self.df_test_val_mix

    def get_dict(self):
        return {
          'train_split': self.df_train_split,
          'test_split': self.df_test_split,
          'train_val': self.df_train_val_mix,
          'test_val': self.df_test_val_mix
        }
  
    def report_train_test_split(self):
        print(f"amount of all data : {self.df_pd.__len__()}")
        print(f"amount of all class : {self.group_df.__len__()}")
        print(f"amount of data 2-8 img : {self.df_2_to_n.__len__()}")
        print(f"amount of 2-8 img class : {self.group_df[(self.group_df['count'] <= self.filter_img) & (self.group_df['count'] > 1)]['labels'].values.__len__()}")
        print(f"amount of data more 8 img : {self.df_more_n.__len__()}")
        print(f"amount of more 8 img class : {self.group_df[self.group_df['count'] > self.filter_img]['labels'].__len__()}")
        print(f"amount of data & class only one : {self.group_df[self.group_df['count'] == 1]['labels'].__len__()}")
        
    def report_train_test_val_split(self):
        print(f"amount of train split : {len(self.df_train_split)}")
        print(f"amount of train split class : {self.df_train_split['labels'].nunique()}")
        print(f"amount of test split : {len(self.df_test_split)}")
        print(f"amount of test split class : {self.df_test_split['labels'].nunique()}")
        print(f"amount of train val : {len(self.df_train_val)}")
        print(f"amount of train val class : {self.df_train_val['labels'].nunique()}")
        print(f"amount of test val : {len(self.df_test_val)}")
        print(f"amount of test val class : {self.df_test_val['labels'].nunique()}")
        print(f"amount of train val mix : {len(self.df_train_val_mix)}")
        print(f"amount of train val mix class : {self.df_train_val_mix['labels'].nunique()}")
        print(f"amount of test val mix : {len(self.df_test_val_mix)}")
        print(f"amount of test val mix class : {self.df_test_val_mix['labels'].nunique()}")