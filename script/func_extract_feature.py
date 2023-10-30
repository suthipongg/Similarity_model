import os
from pathlib import Path

from tqdm.notebook import tqdm

from PIL import Image

import numpy as np
import pandas as pd


class convert_feature:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def process_extract(self, img):
        output = self.pipeline.extract(img)
        return output
    
    def check_data(self, img_path, classes):
        if len(img_path) != len(classes):
            raise ValueError("img_path and classes must have the same length")
    
    def to_pandas(self, output, classes, path):
        df_x = pd.DataFrame(output)
        df_y = pd.DataFrame([[classes, path]], columns=['classes', 'path_img'])
        return pd.concat([df_x, df_y], axis=1)
    
    def save_data(self, output, classes, path, file_name_output, i):
        if len(Path(file_name_output).parents) > 1:
            path_output = Path(file_name_output).parent
            file_name_output = str(Path(file_name_output).name)
        else:
            path_output = ROOT_NFS_TEST
        path_output = path_output / "feature_map"
        
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        path_output = path_output / str(file_name_output+'.csv')
        
        data = self.to_pandas(output, classes, path)
        if i == 0:
            data.to_csv(path_output, index=False)
        else:
            data.to_csv(path_output, mode='a', header=False, index=False)
    
    def __call__(self, df_img_path, df_classes, file_name_output=None, return_extract=False):
        self.check_data(df_img_path, df_classes)
        
        for i, path in enumerate(tqdm(df_img_path)):
            img = Image.open(path).convert('RGB')
            output = self.process_extract(img)
            
            if file_name_output:
                self.save_data(output, df_classes[i], path, file_name_output, i)
            
            if return_extract:
                if i == 0:
                    X_trans = output
                else:
                    X_trans = np.concatenate((X_trans, output))
                    
        if return_extract: 
            return X_trans