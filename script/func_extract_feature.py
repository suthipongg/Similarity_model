import os, time
from pathlib import Path

from tqdm.notebook import tqdm

from PIL import Image

import numpy as np
import pandas as pd
from script.tool import ROOT_NFS_TEST, standardize_feature

from timm.models import create_model
import timm

from onnxruntime import InferenceSession

def select_timm_model(model, num_classes=0, pretrain=True):
    model = create_model(model, num_classes=num_classes, pretrained=pretrain)
    data_config = timm.data.resolve_model_data_config(model)
    processor = timm.data.create_transform(**data_config, is_training=False)
    return model, processor

# pipeline for timm library
class pipeline_timm:
    def __init__(self, device='cuda:0'):
        self.device = device
    
    def selct_model(self, model, processor):
        self.model = model
        self.processor = processor
        self.model.eval().to(self.device)
    
    def process_model(self, img):
        inputs = self.processor(img).to(self.device).unsqueeze(0)
        outputs = self.model(inputs)
        return outputs
        
    def extract(self, img):
        ### return specific layer
        outputs = self.process_model(img)
        outputs.flatten().unsqueeze(0)
        outputs = standardize_feature(outputs).to('cpu').detach().numpy()
        return outputs
    
    def report_test(self):
        img = Image.new('RGB', (224, 224))
        start_time_torch = time.time()
        outputs = self.process_model(img)
        delta_time_torch = time.time() - start_time_torch
        print("runtime :", delta_time_torch*1000, "ms")
        print(f"Output shape at layer : {outputs.shape}")


def select_transformers_model(model, processor, pretrain="google/vit-base-patch16-224-in21k"):
    model = model.from_pretrained(pretrain)
    processor = processor.from_pretrained(pretrain)
    return model, processor

# pipeline for transformer library
class pipeline_transformer:
    def __init__(self, layer, row=False, device='cuda:0'):
        self.device = device
        self.layer = layer
        self.row = row
    
    def selct_model(self, model, processor):
        self.model = model
        self.processor = processor
        self.model.eval().to(self.device)
    
    def process_model(self, img):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs
        
    def extract(self, img):
        ### return specific layer
        outputs = self.process_model(img)
        if type(self.row) == bool and not self.row:
            outputs = outputs[self.layer]
        else:
            outputs = outputs[self.layer][:, self.row]
        outputs = outputs.flatten().unsqueeze(0)
        outputs = standardize_feature(outputs).to('cpu').detach().numpy()
        return outputs
    
    def report_test(self):
        img = Image.new('RGB', (224, 224))
        start_time_torch = time.time()
        outputs = self.process_model(img)
        delta_time_torch = time.time() - start_time_torch
        print("runtime :", delta_time_torch*1000, "ms")
        print(f"outputs layers : {outputs.keys()}")
        print(f"shape last_hidden_state : {outputs.last_hidden_state.shape}")
        print(f"shape pooler_output : {outputs.pooler_output.shape}")


def select_transformers_onnx_model(path="google/vit-base-patch16-224-in21k", processor=None, providers=['CPUExecutionProvider']):
    model = InferenceSession(path, providers=providers)
    processor = processor.from_pretrained(Path(path).parent)
    return model, processor

# pipeline for transformer onnx library
class pipeline_transformer_onnx:
    def __init__(self, layer, row=False):
        self.layer = layer
        self.row = row
    
    def selct_model(self, model, processor):
        self.model = model
        self.processor = processor
    
    def process_model(self, img):
        inputs = self.processor(images=img, return_tensors="np")
        outputs = self.model.run(output_names=[self.layer], input_feed=dict(inputs))[0]
        return outputs
        
    def extract(self, img):
        ### return specific layer
        outputs = self.process_model(img)
        if type(self.row) == bool and not self.row:
            outputs = outputs[0]
        else:
            outputs = outputs[:, self.row]
        outputs = standardize_feature(outputs)
        return outputs
    
    def report_test(self):
        img = Image.new('RGB', (224, 224))
        start_time_torch = time.time()
        outputs = self.process_model(img)
        delta_time_torch = time.time() - start_time_torch
        print("runtime :", delta_time_torch*1000, "ms")
        print(f"shape : {outputs.shape}")


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