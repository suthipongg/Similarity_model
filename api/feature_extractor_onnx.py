import onnxruntime as rt
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from config import ConFig

arg = ConFig()

class FeatureExtractor:
    def __init__(self):
        self.device = rt.get_device()
        self.output_name = ["avg_pool"]
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'GPU' else ['CPUExecutionProvider']
        self.sess = rt.InferenceSession(arg.PATH_ONNX_EMBEDDED, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape[1:3]
        

        dummy_img = Image.new('RGB', size=self.input_shape)
        dummy_img = self.preprocess(dummy_img)
        _ = self.sess.run(self.output_name, {self.input_name: dummy_img})
        print('Initial Run Pass ...')

      
    def preprocess(self, img):
        img = img.resize(tuple(self.input_shape))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)   
        return x  


    def extract(self, img):
        x = self.preprocess(img)
        feature = self.sess.run(self.output_name, {self.input_name: x})[0]
        feature = feature / np.linalg.norm(feature)  # Normalize
        return feature[0]


if __name__ == '__main__':
    from PIL import Image
    from pathlib import Path


    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        print(feature)