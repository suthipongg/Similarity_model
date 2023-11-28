from elasticsearch import Elasticsearch
from tqdm.notebook import tqdm

import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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
                "mappings":
                {
                    "properties":
                    {
                        "id": {"type":"keyword"},
                        "tag": {"type":"keyword"},
                        "labels": {"type":"keyword"},
                        "file_names": {"type":"text"},
                        "images_path": {"type":"text"},
                        "bid": {"type":"keyword"},
                        "scid": {"type":"keyword"},
                        "cid": {"type":"keyword"},
                        "action": {"type":"keyword"},
                        "features": 
                        {  
                            "type":"dense_vector",
                            "dims":dims,
                            "index":True,
                            "similarity": "dot_product"
                        }
                    }
                }
            }
            err = self.es.indices.create(index=self.name_index, body=body_product)
            print(err)
            return False

    def find_top_n(self, id, tag_name_compare, top_n=5, collapse=True, filter_all=None, use_production=False):
        product = self.es.get(index=self.name_index, id=id)['_source']
        label_true = product['labels']
        if filter_all is not None:
            filter_all = {}
            for key in ['bid', 'scid', 'cid']:
                if product[key] > -1:
                    num = -2
                    if key == 'bid': num = -4
                    filter_all[key] = [product[key], num]

        body_query = query_cosine(product['features'], tag_name_compare, top_n, 
                                  collapse, filter_all, use_production)
        result = self.es.search(index=self.name_index,doc_type=self.name_doc,
                                body=body_query)['hits']['hits']
        return result, product

        
class extract_to_es(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)

    def check_data_exist(self, data, n):
        if self.es.exists(index=self.name_index, id=data['tag']+"_"+str(n)):
            data_index = self.es.get(index=self.name_index, id=data['tag']+"_"+str(n))['_source']
            for key in ['tag', 'labels', 'file_names', 'images_path', 'id']:
                if data[key] != data_index[key]:
                    print(f"{data['tag']+'_'+str(n)} | key {key} not match | data in es -> {data_index[key]} != new data -> {data[key]}")
                    return False
            return True
        return False
    
    def put_to_es(self, model, dataframe, tag="train_split", replace=True, crop=False):
        for n, img_path in enumerate(tqdm(dataframe['images_path'], leave=False, desc=tag)):
            data = {
                "id": tag+"_"+str(n),
                "tag": tag,
                "labels": dataframe['labels'].iloc[n],
                "file_names": dataframe['file_names'].iloc[n],
                "images_path": img_path,
                "bid": dataframe['BID'].iloc[n],
                "scid": dataframe['SCID'].iloc[n],
                "cid": dataframe['CID'].iloc[n],
                "action": dataframe['Action'].iloc[n]
            }
            if not replace and self.check_data_exist(data, n):
                continue
            img = Image.open(img_path).convert('RGB')
            output = model.extract(img).flatten()
            data["features"] = to_unit_len(output)
            self.es.index(index=self.name_index, id=tag+"_"+str(n), body=data)
        print(f"put tag {tag} success")

    def put_all_tag(self, model, df, replace=True, crop=False):
        self.put_to_es(model, df["train_split"], tag="train_split", replace=replace, crop=crop)
        self.put_to_es(model, df["test_split"], tag="test_split", replace=replace, crop=crop)
        self.put_to_es(model, df["train_val"], tag="train_val", replace=replace, crop=crop)
        self.put_to_es(model, df["test_val"], tag="test_val", replace=replace, crop=crop)
        self.put_to_es(model, df["one_img"], tag="one_img", replace=replace, crop=crop)


class measure_score(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)

    def accuracy(self, tag_names, tag_name_compare, top_n=5, filter_all=None):
        tag_names = to_list(tag_names)
        list_scores = np.array([0] * top_n)
        counter = 0
        for tag_name in tag_names:
            count_res = self.es.count(index=self.name_index,doc_type=self.name_doc,
                                      body=query_tag_count(tag_name))['count']
            counter += count_res
            for i in tqdm(range(count_res), desc=str(tag_name), leave=False):
                result, product = self.find_top_n(tag_name+"_"+str(i), tag_name_compare, top_n, filter_all)
                label_true = product['labels']
                for top, row in enumerate(result):
                    if label_true == row['_source']['labels']:
                        list_scores[top:] += 1
                        break

        list_scores = (list_scores/counter*100).round(2)
        return list_scores

    def report_acc(self, top_n=5, filter_all=None):
        result = []
            
        list_scores_test_val = self.accuracy(tag_names='test_val', tag_name_compare="train_val", top_n=top_n, 
            filter_all=filter_all)
        result.append(list_scores_test_val)
        print(f"validate score : {list_scores_test_val}")

        list_scores_test_split = self.accuracy(tag_names='test_split', tag_name_compare="train_split", 
            top_n=top_n, filter_all=filter_all)
        result.append(list_scores_test_split)
        print(f"split score : {list_scores_test_split}")

        list_scores_test_split_val = self.accuracy(tag_names=['test_val', 'test_split'], 
            tag_name_compare=['train_val', "train_split"], top_n=top_n, filter_all=filter_all)
        result.append(list_scores_test_split)
        print(f"split+val score : {list_scores_test_split_val}")

        return result

# Show image
class report_image(ES_access):
    def __init__(self, name_index, name_doc='_doc', url="http://localhost:9200"):
        super().__init__(name_index, name_doc, url)
    
    def show_image(self, axes, img_path, row, col, title, img=None):
        if img is None:
            img = Image.open(img_path).convert('RGB')
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(title)
    
    def plot_image(self, list_id, tag_name_compare=['train_split', 'train_val', 'one_img'], 
                   top_n=5, collapse=True, filter_all=None, use_production=False):
        list_id = to_list(list_id)
        rows = len(list_id)
        columns = top_n+1
        fig, axes = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
        if len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
            
        for row, id in enumerate(list_id):
            result, product = self.find_top_n(id, tag_name_compare, top_n, collapse, filter_all, use_production)
            self.show_image(axes, product['images_path'], row, 0, str(product['labels']) + " : True")
            for col, res in enumerate(result):
                score = res['_score']
                res = res['_source']
                self.show_image(axes, res['images_path'], row, 
                                col+1, str(res['labels']) + " : " + str(round(score*100, 2)))
            for n in range(len(result)+1, top_n+1):
                self.show_image(axes, res['images_path'], row, n, "", img=np.array([[[255, 255, 255]]]))
        plt.show()
        
    def random_id(self, tag_name, num_product=5):
        count_res = self.es.count(index=self.name_index, doc_type=self.name_doc, 
                                  body=query_tag_count(tag_name))['count']
        random_id = random.sample(range(0, count_res), num_product)
        return [tag_name + "_" + str(id) for id in random_id]
    
    def show_report(self, num_product=5, top_n=5, collapse=True, filter_all=None, 
                    use_production=False, list_id=None):
        if list_id is None:
            list_split = self.random_id("test_split", num_product)
            list_val = self.random_id("test_val", num_product)
        else:
            list_split = list_id['split']
            list_val = list_id['val']
        list_split_val = list_split + list_val
        
        print("Split :")
        self.plot_image(list_split, "train_split", top_n, collapse, filter_all, use_production)
        
        print()
        print("=================================================================================================")
        print()
        
        print("Validate :")
        self.plot_image(list_val, "train_val", top_n, collapse, filter_all, use_production)

        print()
        print("=================================================================================================")
        print()
        
        print("Split + Validate :")
        self.plot_image(list_split_val, ['train_split', 'train_val'], top_n, collapse, filter_all, use_production)
        return {
            'split': list_split, 
            'val': list_val
        }