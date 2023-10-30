from prettytable import PrettyTable
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class split_StratifiedKFold(StratifiedKFold):
    def __init__(self, n_cv=5, shuffle=True, random_state=42):
        super().__init__(n_splits=n_cv, shuffle=shuffle, random_state=random_state)
        self.n_cv = n_cv

    def filter_data(self):
        counts = self.df_y['classes'].value_counts()
        classes_less_than_n = counts[counts < self.n_cv].index
        self.index_less = self.df_y['classes'].isin(classes_less_than_n)
        self.index_greater = ~self.df_y['classes'].isin(classes_less_than_n)
        print(f"amount of image that less than {self.n_cv} in that class : {sum(self.index_less)}")
        print(f"amount of image that more than {self.n_cv} in that class : {sum(self.index_greater)}")

    def split_skf(self, X, y):
        self.df_y = y
        self.filter_data()
        self.X_less = X[self.index_less]
        self.df_y_less = self.df_y[self.index_less]
        self.X_greater = X[self.index_greater]
        self.df_y_greater = self.df_y[self.index_greater]
        
        self.index_df_split = self.split(self.X_greater, self.df_y_greater["classes"])

    def train_test_index_split(self, train_index, test_index):
        x_train_greater = np.array(self.X_greater)[train_index]
        df_y_train_greater = np.array(self.df_y_greater)[train_index]
        x_test_greater = np.array(self.X_greater)[test_index]
        df_y_test_greater = np.array(self.df_y_greater)[test_index]
        
        x_train = np.concatenate((x_train_greater, self.X_less))
        df_y_train = np.concatenate((df_y_train_greater, self.df_y_less))
        return x_train, df_y_train, x_test_greater, df_y_test_greater

    def compute_similarity(self, x_test_greater, x_train):
        dot_product = np.dot(x_test_greater,x_train.T)              # (x_test , x_train)
        norm_test = norm(x_test_greater, axis=1).reshape(-1, 1)     # (x_test, 1)
        norm_train = norm(x_train, axis=1).reshape(1, -1)           # (1, x_train)
        similar_score = dot_product/(norm_test*norm_train)          # res = (x_test , x_train), norm_test*norm_train = (x_test , x_train)
        return similar_score

    def sort_top_n(self, similar_score, df_y_train, n=5):
        ranking = np.argsort(similar_score, axis=1)[:, ::-1]
        for row, rank in enumerate(tqdm(ranking, desc="ranking")):
            df_y_train_sort = df_y_train[rank]
            score = similar_score[row][rank].reshape(-1, 1)
            df_row = pd.DataFrame(np.concatenate((df_y_train_sort, score), axis=1), 
                                columns=['path_img', 'classes', 'score'])
            df_row['score'] = df_row['score'].astype(float)
            idx = df_row.groupby('classes')['score'].idxmax()
            result = df_row.loc[idx]
            result = result.sort_values(by='score', ascending=False)[:n]
            if row == 0:
                self.rank_top_n = [result]
            else:
                self.rank_top_n = np.concatenate((self.rank_top_n, [result]))

    def score_top_n(self):
        return sum((self.df_y_test_greater[:, 1].reshape(-1, 1) == self.rank_top_n[:, :, 1]).any(axis=1))/self.rank_top_n.shape[0]

    def score_top(self):
        return sum(self.df_y_test_greater[:, 1] == self.rank_top_n[:, 0, 1])/self.rank_top_n.shape[0]
    
    def print_table(self, top_score, top_n, n):
        tab = PrettyTable()
        tab.field_names = ["accuracy method"] + ["cv"+str(cv) for cv in range(self.n_cv)] + ["avg"]
        top_score_avg = np.mean(top_score).round(2)
        top_n_avg = np.mean(top_n).round(2)
        tab.add_row(["top score"] + top_score + [top_score_avg])
        tab.add_row([f"top {n}"] + top_n + [top_n_avg])
        print(tab)
        return top_score_avg, top_n_avg

    def report_score(self, n=5):
        result, result_in_n = [], []
        for i, (train_index, test_index) in enumerate(tqdm(self.index_df_split, desc="cv", total=self.n_cv)):
            x_train, df_y_train, x_test_greater, self.df_y_test_greater = self.train_test_index_split(train_index, test_index)

            similar_score = self.compute_similarity(x_test_greater, x_train)
            self.sort_top_n(similar_score, df_y_train, n)
            
            acc_top_n = round(self.score_top_n()*100, 2)
            acc_top = round(self.score_top()*100, 2)
            if i == 0:
                result, result_in_n = [acc_top], [acc_top_n]
            else:
                result.append(acc_top)
                result_in_n.append(acc_top_n)
        return self.print_table(result, result_in_n, n)
    
    def show_image(self, axes, img_path, row, col, title):
        img = Image.open(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(title)
    
    def plot_image(self, list_data):
        rows = len(list_data)
        columns = self.n_cv+1
        fig, axes = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
        test_path = self.df_y_test_greater[list_data, 0]
        test_class = self.df_y_test_greater[list_data, 1]
        train_path = self.rank_top_n[list_data, :, 0]
        train_score = self.rank_top_n[list_data, :, 2]*100
        train_score = train_score.astype(float).round(2)
        train_class = self.rank_top_n[list_data, :, 1]
        for row, test_img in enumerate(test_path):
            self.show_image(axes, test_img, row, 0, str(test_class[row]) + " : test")
            for col, train_img in enumerate(train_path[row]):
                self.show_image(axes, train_img, row, 
                                col+1, str(train_class[row, col]) + " : " + str(train_score[row, col]))
        plt.show()