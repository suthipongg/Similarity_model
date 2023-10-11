from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = EfficientNetB1(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.input_shape = self.model.input.shape[1:3]
        dummpy_img = Image.new("RGB", size=tuple(self.input_shape))
        dummpy_img = self.preprocess(dummpy_img)
        _ = self.model.predict(dummpy_img)[0]
        print('Initial Run Pass ...')

    
    def preprocess(self, img):
        img = img.resize(self.input_shape)  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        return x

    def extract(self, img):
        x = self.preprocess(img)
        feature = self.model.predict(x)[0]  # (1, 1280) -> (1280, )
        return feature / np.linalg.norm(feature)  # Normalize



if __name__=='__main__':
    from PIL import Image

    fe = FeatureExtractor()
    img_path = r'testCode\static\uploaded\2022-03-07T15.12.26.251485_snailwhite-body-booster-1.jpg'
    feature = fe.extract(img=Image.open(img_path))
    print(feature)