from elasticsearch import Elasticsearch
from tqdm.notebook import tqdm

import random
import matplotlib.pyplot as plt
from PIL import Image

from script.func_query_body import to_list, query_tag, query_tag_count, query_cosine

# Elasticsearch accesss compute score
class ES_access:
    def __init__(self, name_index, name_doc, url="http://localhost:9200"):
        self.es = Elasticsearch(url)
        self.name_index = name_index
        self.name_doc = name_doc

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
        return list_split, list_val