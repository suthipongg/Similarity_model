from script.tool import ROOT_NFS_TEST, standardize_feature
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


class PlotPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=3)
        
    def transform_data(self, data_name, filter_classes_num=10, data_path=ROOT_NFS_TEST / "feature_map"):
        self.df_pd = pd.read_csv(data_path / data_name)
        self.filter_classes = self.df_pd['classes'].unique()[:filter_classes_num]
        index_filter_class = self.df_pd["classes"].isin(self.filter_classes)
        X = self.df_pd.loc[index_filter_class].iloc[:, :-1].values
        X = standardize_feature(X)
        y = self.df_pd.loc[index_filter_class].iloc[:, -1].values
        self.reduced_data = self.pca.fit_transform(X)
        
    def get_explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_
    
    def plot_3d(self):
        x_plot = self.reduced_data[:, 0]
        y_plot = self.reduced_data[:, 1]
        z_plot = self.reduced_data[:, 2]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for g in self.filter_classes:
            ix = np.where(self.df_pd['classes'] == g)
            ax.scatter(x_plot[ix], y_plot[ix], z_plot[ix], label = g, s = 100)
        ax.legend()
        plt.show()