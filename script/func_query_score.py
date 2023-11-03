from elasticsearch import Elasticsearch
from tqdm.notebook import tqdm

import random
import matplotlib.pyplot as plt
from PIL import Image

from script.func_query_body import to_list, query_tag, query_tag_count, query_cosine
from script.tool import to_unit_len

# Elasticsearch accesss compute score
class ES_access:
    def __init__(self, name_index, name_doc="_doc", url="http://localhost:9200"):
        self.es = Elasticsearch(url)
        self.name_index = name_index
        self.name_doc = name_doc
        
    def check_index_exist(self, dims=768):
        if self.es.indices.exists(index=self.name_index):
            print(f"index {self.name_index} already exists")
            return True
        else:
            body_product = {
                "mappings":{
                    "properties":{
                        "tag":{
                            "type":"keyword"
                        },
                        "labels":{
                            "type":"keyword"
                        },
                        "file_names":{
                            "type":"text"
                        },
                        "images_path":{
                            "type":"text"
                        },
                        "features":{  
                            "type":"dense_vector",
                            "dims":dims,
                            "index":True,
                            "similarity": "dot_product"
                        },
                        "id":{
                            "type":"keyword"
                        }
                    }
                }
            }
            err = self.es.indices.create(index=self.name_index, body=body_product)
            print(err)
            return False

class extract_to_es(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)
        self.load_crop = False

    def check_data_exist(self, data, n):
        if self.es.exists(index=self.name_index, id=data['tag']+"_"+str(n)):
            data_index = self.es.get(index=self.name_index, id=data['tag']+"_"+str(n))['_source']
            for key in ['tag', 'labels', 'file_names', 'images_path', 'id']:
                if data[key] != data_index[key]:
                    print(f"{data['tag']+"_"+str(n)} | key {key} not match | data in es -> {data_index[key]} != new data -> {data[key]}")
                    return False
            return True
        return False
    
    def put_to_es(self, model, dataframe, tag="train_split", replace=True, crop=False):
        for n, img_path in enumerate(tqdm(dataframe['images_path'], leave=False)):
            data = {
                "tag": tag,
                "labels": dataframe['labels'].iloc[n],
                "file_names": dataframe['file_names'].iloc[n],
                "images_path": img_path,
                "id": tag+"_"+str(n)
            }
            if not replace and self.check_data_exist(data, n):
                continue
            img = Image.open(img_path).convert('RGB')
            if crop:
                if not self.load_crop:
                    from script.func_crop import CropProduct
                    self.load_crop = True
                    self.detector = CropProduct()
                img = self.detector.detect(img, thresh=0.5)
            output = model.extract(img).flatten()
            data["features"] = to_unit_len(output)
            self.es.index(index=self.name_index, id=tag+"_"+str(n), body=data)
        print(f"put tag {tag} success")

    def put_all_tag(self, model, df, replace=True, crop=False):
        self.put_to_es(model, df['train_split'], tag="train_split", replace=replace, crop=crop)
        self.put_to_es(model, df['test_split'], tag="test_split", replace=replace, crop=crop)
        self.put_to_es(model, df['train_val'], tag="train_val", replace=replace, crop=crop)
        self.put_to_es(model, df['test_val'], tag="test_val", replace=replace, crop=crop)

class measure_score(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)

    def accuracy(self, tag_names, tag_name_compare, top_n=5):
        tag_names = to_list(tag_names)
        score_top = 0
        score_top_n = 0
        counter = 0
        for tag_name in tag_names:
            count_res = self.es.count(index=self.name_index,doc_type=self.name_doc,body=query_tag_count(tag_name))['count']
            counter += count_res
            for i in tqdm(range(count_res), desc=str(tag_name), leave=False):
                product = self.es.get(index=self.name_index, id=tag_name+"_"+str(i))['_source']
                label_test = product['labels']
                result = self.es.search(index=self.name_index,doc_type=self.name_doc,body=query_cosine(product['features'], tag_name_compare, top_n))['hits']['hits']
                # pred = []
                label_predict = []
                for row in result:
                    row['_source']['_score'] = row['_score']
                    # pred.append(row['_source'])
                    label_predict.append(row['_source']['labels'])
                if label_test == label_predict[0]:
                    score_top += 1
                if label_test in label_predict:
                    score_top_n += 1
        score_top = score_top/counter*100
        score_top_n = score_top_n/counter*100
        return score_top, score_top_n

    def report_acc(self):
        result = []
        
        score_test_val_top, score_test_val_top_n = self.accuracy(
            tag_names='test_val', tag_name_compare="train_val", top_n=5)
        result.append([score_test_val_top, score_test_val_top_n])
        print(f"validate top score : {round(score_test_val_top, 2)} , validate top 5 score : {round(score_test_val_top_n, 2)}")

        score_test_split_top, score_test_split_top_n = self.accuracy(
            tag_names='test_split', tag_name_compare="train_split", top_n=5)
        result.append([score_test_split_top, score_test_split_top_n])
        print(f"split top score : {round(score_test_split_top, 2)} , split top 5 score : {round(score_test_split_top_n, 2)}")

        score_test_split_val_top, score_test_split_val_top_n = self.accuracy(
            tag_names=['test_val', 'test_split'], tag_name_compare=['train_val', "train_split"], top_n=5)
        result.append([score_test_split_val_top, score_test_split_val_top_n])
        print(f"split+val top score : {round(score_test_split_val_top, 2)} , split+val top 5 score : {round(score_test_split_val_top_n, 2)}")

        return result

# Show image
class report_image(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)
    
    def show_image(self, axes, img_path, row, col, title):
        img = Image.open(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(title)

    def find_top_n(self, feature, tag_name_compare, top_n, collapse):
        pred = []
        result = self.es.search(index=self.name_index, doc_type=self.name_doc, 
                           body=query_cosine(feature, tag_name_compare, top_n, collapse))['hits']['hits']
        for row in result:
            row['_source']['_score'] = row['_score']
            pred.append(row['_source'])
        return pred
    
    def plot_image(self, list_id, top_n, tag_name_compare, collapse):
        rows = len(list_id)
        columns = top_n+1
        fig, axes = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))

        for row, id in enumerate(list_id):
            product = self.es.get(index=self.name_index, id=id)['_source']
            self.show_image(axes, product['images_path'], row, 0, str(product['labels']) + " : test")
            pred = self.find_top_n(product['features'], tag_name_compare, top_n, collapse)
            for col, pred_dict in enumerate(pred):
                self.show_image(axes, pred_dict['images_path'], row, 
                                col+1, str(pred_dict['labels']) + " : " + str(round(pred_dict['_score']*100, 2)))
        plt.show()
        
    def random_id(self, tag_name, num_product=5):
        count_res = self.es.count(index=self.name_index, doc_type=self.name_doc, body=query_tag_count(tag_name))['count']
        random_id = random.sample(range(0, count_res), num_product)
        return [tag_name + "_" + str(id) for id in random_id]
    
    def show_report(self, num_product=5, top_n=5, collapse=True, list_id=None):
        if list_id is None:
            list_split = self.random_id("test_split", num_product)
            list_val = self.random_id("test_val", num_product)
        else:
            list_split = list_id['split']
            list_val = list_id['val']
        list_split_val = list_split + list_val
        
        print("Split :")
        self.plot_image(list_split, top_n, "train_split", collapse)
        
        print()
        print("======================================================================================================================")
        print()
        
        print("Validate :")
        
        self.plot_image(list_val, top_n, "train_val", collapse)

        print()
        print("======================================================================================================================")
        print()
        
        print("Split + Validate :")
        
        self.plot_image(list_split_val, top_n, ['train_split', 'train_val'], collapse)
        return {'split': list_split, 
                'val': list_val}