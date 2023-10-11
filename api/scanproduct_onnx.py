from flask import Blueprint, request, jsonify, json
from config import ConFig
from PIL import Image, ImageOps
import sys
import os
from .feature_extractor_onnx import FeatureExtractor
from .detection_yolov7 import ProductDetector
import logging
import numpy as np

LOG = logging.getLogger(__name__)

scanproduct_onnx_api = Blueprint('scanproduct_onnx_api', __name__)
fe = FeatureExtractor()
detector = ProductDetector()

arg = ConFig()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in arg.ALLOWED_EXTENSIONS


@scanproduct_onnx_api.route('/cosmenet/scanproduct/test', methods=["GET", "POST"])
def index():
    LOG.info('test successfully')    # LOG
    return jsonify({'message': 'successfully'})



@scanproduct_onnx_api.route('/cosmenet/scanproduct/v2/onnx', methods=["POST", "GET"])
def getfullpath():
    if request.method == "POST":
        re = request.get_json()
        fullPath = re["fullPath"]
        print(f'fullPath : {fullPath}')
        LOG.info(f'fullPath : {fullPath}' )    # LOG

        try:
            local_fullPath = os.path.join(arg.PATH_DIRECTORY_LOCAL, os.path.join(*fullPath.split('/')[3:]))
            print(f'local_fullPath : {local_fullPath}')
            LOG.info(f'local_fullPath : {local_fullPath}' )    # LOG
            if os.path.exists(local_fullPath):

                errors = {}
                output = {}
                success = False
                if allowed_file(fullPath):
                    img = Image.open(local_fullPath)
                    img = ImageOps.exif_transpose(img)
                    img = img.convert('RGB')
                    output = detector.detector(img)
                    # # Save the image to a file
                    # Image.fromarray(output['img']).save("output.jpg")
                    # # 
                    embedded_feature = fe.extract(Image.fromarray(output['img'])).tolist()
                    success = True
                else:
                    LOG.info('File type is not allowed')    # LOG
                    errors[fullPath] = 'File type is not allowed'

                if success and errors:
                    LOG.info('File(s) successfully scanned')    # LOG
                    print('File(s) successfully scanned')
                    errors['message'] = 'File(s) successfully scanned'
                    resp = jsonify(errors)
                    return resp, 200

                if success:
                    LOG.info('Files successfully scanned')    # LOG
                    print('Files successfully scanned')
                    resp = json.dumps({'message': 'Files successfully scanned',
                                        'embedded_feature': embedded_feature,
                                        'score': round(float(output['score']), 2),
                                        'bbox': output['bbox']
                                        })
                    return resp, 200
                else:
                    print(errors)
                    resp = jsonify(errors)
                    return resp, 200
            
            else:
                LOG.info('No file part in Directory')    # LOG
                print('No file part in Directory')
                resp = jsonify({'message': 'No file part in Directory'})
                return resp, 200

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)
            LOG.info('Path is not correct')    # LOG
            print('Path is not correct')
            resp = jsonify({'message': 'Path is not correct'})
            return resp, 200

    else:
        LOG.info('method GET api scan product')    # LOG
        return jsonify({'message': 'method GET api scan product'})
            


@scanproduct_onnx_api.route('/cosmenet/scanproduct/backend/onnx', methods=["POST", "GET"])
def getfullpath_backend():
    if request.method == "POST":
        re = request.get_json()
        fullPath = re["fullPath"]
        print(f'fullPath : {fullPath}')
        LOG.info(f'fullPath : {fullPath}' )    # LOG

        try:
            local_fullPath = os.path.join(arg.PATH_DIRECTORY_LOCAL_BACKEND, os.path.join(*fullPath.split('/')[3:]))
            print(f'local_fullPath : {local_fullPath}')
            LOG.info(f'local_fullPath : {local_fullPath}' )    # LOG
            if os.path.exists(local_fullPath):

                errors = {}
                success = False
                if allowed_file(fullPath):
                    img = Image.open(local_fullPath)
                    img = ImageOps.exif_transpose(img)
                    img = img.convert('RGB')
                    embedded_feature = fe.extract(img).tolist()
                    success = True
                else:
                    LOG.info('File type is not allowed')    # LOG
                    errors[fullPath] = 'File type is not allowed'

                if success and errors:
                    LOG.info('File(s) successfully scanned')    # LOG
                    print('File(s) successfully scanned')
                    errors['message'] = 'File(s) successfully scanned'
                    resp = jsonify(errors)
                    return resp, 200

                if success:
                    LOG.info('Files successfully scanned')    # LOG
                    print('Files successfully scanned')
                    resp = json.dumps({'message': 'Files successfully scanned',
                                        'embedded_feature': embedded_feature,
                                        })
                    return resp, 200
                else:
                    print(errors)
                    resp = jsonify(errors)
                    return resp, 200
            
            else:
                LOG.info('No file part in Directory')    # LOG
                print('No file part in Directory')
                resp = jsonify({'message': 'No file part in Directory'})
                return resp, 200

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, e, fname, exc_tb.tb_lineno)
            LOG.info('Path is not correct')    # LOG
            print('Path is not correct')
            resp = jsonify({'message': 'Path is not correct'})
            return resp, 200

    else:
        LOG.info('method GET api backend scan product')    # LOG
        return jsonify({'message': 'method GET api scan product'})