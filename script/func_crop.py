import numpy as np
from PIL import Image, ImageOps
import onnxruntime as rt

class CropProduct:
    def __init__(self):
        device = rt.get_device()
        w = '/app/nfs_clientshare/mew/project/Similarity_model/models/yolov7-cosme.onnx'
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'GPU' else ['CPUExecutionProvider']
        self.session = rt.InferenceSession(w, providers=providers)
        
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

    def add_border(image, border_color=(0, 0, 0), target_size=640):
        # # Load the image
        # image = Image.open(image_path)
    
        # Get the original image size
        width, height = image.size
    
        # Scale ratio (new / old)
        ratio = min(target_size / height, target_size / width)
    
        # Resize the image to a 640x640 ratio based on the old width or height
        if width >= height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        resized_image = image.resize((new_width, new_height))
    
        # Add a border to the resized image
        border_size = (int((target_size - new_width) / 2), int((target_size - new_height) / 2))
        bordered_image = ImageOps.expand(resized_image, border=border_size, fill=border_color)
        bordered_image = bordered_image.resize((target_size, target_size))
    
        # Return the bordered image
        return bordered_image, ratio, border_size

    def preprocess_yolo(img, target_size=640):
        image, ratio, dwdh = self.add_border(img, target_size=target_size)
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        return im, ratio, dwdh
    
    def detect(img, input_size=640, target_size=224, thresh=0.9):
        im, ratio, dwdh = self.preprocess_yolo(img, target_size=input_size)
        img = np.array(img)
        ori_images = [img.copy()]
    
        inp = {self.inname[0]:im}
    
        # ONNX inference
        outputs = self.session.run(self.outname, inp)[0]
    
        if len(outputs) != 0:
            for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                if score < thresh or i > 0:
                    return {
                        'img': ori_images[int(batch_id)],
                        'score': score,
                        'bbox': None
                    }
                
                image = ori_images[int(batch_id)]
                # Get the image size of the image
                h_image, w_image, channels = image.shape
            
                box = np.array([x0,y0,x1,y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
    
                # Limit the bounding box coordinates to the image bounds
                box[0] = max(0, min(box[0], w_image))
                box[1] = max(0, min(box[1], h_image))
                box[2] = max(0, min(box[2], w_image))
                box[3] = max(0, min(box[3], h_image))
    
                cls_id = int(cls_id)
                score = round(float(score),3)
                img_crop = image[box[1]:box[3], box[0]:box[2]]
                img_crop = Image.fromarray(img_crop['img'])
    
            return {
                'img': self.add_border(img_crop, target_size=target_size)[0],
                'score': score,
                'bbox': box
            }
        else:
            return {
                        'img': self.add_border(Image.fromarray(ori_images[0]), target_size=target_size)[0],
                        'score': 0.0,
                        'bbox': None
                    }