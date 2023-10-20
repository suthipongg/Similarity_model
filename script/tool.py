import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def get_output_at_layer(model, layer_name, input_data):
    # Ensure the model is in evaluation mode
    model.eval()

    # Register hooks to capture intermediate activations
    activations = {}
    def hook(module, input, output):
        activations[layer_name] = output
    hook_handle = model._modules[layer_name].register_forward_hook(hook)

    # Forward pass the input data
    with torch.no_grad():
        output = model(input_data)

    # Remove the hook
    hook_handle.remove()

    return activations[layer_name]


blacklist = [".DS_Store", "บาร์โค้ด", "barcode", "desktop.ini", "train_data.csv"]
def in_blacklist(file):
    for bl in blacklist:
        if bl.lower() in file.lower():
            return True
    return False


def scan_directory(path):
    # Initialize a list to store the paths of JPG files
    df = []

    # Use os.walk to traverse the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if not in_blacklist(file):
                df.append([os.path.join(root, file), os.path.basename(root)])
    return np.array(df)

def filter_data(data, minimum_data_class=5):
    df_pd = pd.DataFrame(data, columns = ['path_img','classes'])
    df_pd['classes_labeled'], _ = pd.factorize(df_pd['classes'])
    counts = df_pd['classes_labeled'].value_counts()
    classes_less_than_n = counts[counts < minimum_data_class].index
    index_less_than_n = df_pd['classes_labeled'].isin(classes_less_than_n)
    index_greater_than_or_equal_to_n = ~df_pd['classes_labeled'].isin(classes_less_than_n)
    print("amount of all image :", len(df_pd))
    print(f"amount of image that less than {minimum_data_class} in that class : {sum(index_less_than_n)}")
    print(f"amount of image that more than {minimum_data_class} in that class : {sum(index_greater_than_or_equal_to_n)}")
    return df_pd, index_less_than_n, index_greater_than_or_equal_to_n

def standardize_feature(arr):
    return (arr-arr.mean())/arr.std()

def feature_extract(model, processor, pretain, df_path_img, device='cpu'):
    X_trans = []
    first = True
    processor = processor.from_pretrained(pretain)
    for img_path in tqdm(df_path_img):
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        output_at_layer = outputs.last_hidden_state[:, 0]
        output_at_layer = output_at_layer.flatten().unsqueeze(0)
        output_at_layer = standardize_feature(output_at_layer).to('cpu').numpy()
        if first:
            X_trans = output_at_layer
            first = False
        else:
            X_trans = np.concatenate((X_trans, output_at_layer))
    
    return X_trans

def save_feature(X, y, name="result", path=ROOT):
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=['classes'])
    data = pd.concat([df_x, df_y], axis=1)
    data.to_csv(Path(path) / 'feature_map' / str(name+'.csv'), index=False)

def load_feature(file_name, path=ROOT):
    file_path = path / 'feature_map' / file_name
    df = pd.read_csv(file_path)
    X = np.array(df.iloc[:, :-1])
    y = np.array(df['classes'])
    return X, y

def StratifiedKFold_score(X, y, n_cv=5, index_filter = False):
    first = True
    result = []
    result_in_n = []
    skf = StratifiedKFold(n_splits=n_cv)
    if index_filter != False:
        index_greater_filtered, index_less_filtered = index_filter
        X_less = X[index_less_filtered]
        y_less = y[index_less_filtered]
        X = X[index_greater_filtered]
        y = y[index_greater_filtered]
                
    index_df_split = skf.split(X, y)

    for train_index, test_index in index_df_split:
        x_train = np.array(X)[train_index]
        y_train = np.array(y)[train_index]
        x_test = np.array(X)[test_index]
        y_test = np.array(y)[test_index]
        
        if index_filter != False:
            x_train = np.concatenate((x_train, X_less))
            y_train = np.concatenate((y_train, y_less))

        dot_product = np.dot(x_test,x_train.T)              # (x_test , x_train)
        norm_test = norm(x_test, axis=1).reshape(-1, 1)     # (x_test, 1)
        norm_train = norm(x_train, axis=1).reshape(1, -1)   # (1, x_train)
        res = dot_product/(norm_test*norm_train)            # res = (x_test , x_train), norm_test*norm_train = (x_test , x_train)
        
        y_pred = y_train[np.argmax(res, axis=1)]
        acc = accuracy_score(y_test, y_pred)
        if first:
            first = False
            result = [acc]
        else:
            result.append(acc)
    return result, [train_index, test_index], res

def plot_image(n, img_path, ls, res_t, columns = 6, add_row=False):
    rows = len(n)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 2 * rows))
    train_path = np.array(img_path.iloc[ls[0]])
    if type(add_row) != bool:
        less_than_n = np.array(add_row)
        train_path = np.concatenate((train_path, less_than_n))
    else:
        train_path = np.array(img_path.iloc[ls[0]])
    test_path = np.array(img_path.iloc[ls[1]])
    ind_top_n = np.argsort(res_t, axis=1)[:, -(columns-1):]
    for n_row, row in enumerate(n):
        i_path = [test_path[row]]
        for i in range(-1, -columns, -1):
            ind = ind_top_n[row][i]
            i_path.append(train_path[ind])

        scores = res_t[n_row][ind_top_n[n_row]]
        for i, image_path in enumerate(i_path):
            img = Image.open(image_path)
            axes[n_row, i].imshow(img)
            axes[n_row, i].axis('off')
            t1 = str(Path(image_path).name.split("_")[0])
            if i == 0:
                t2 = "test"
            else:
                score = scores[columns - 1 - i]
                t2 = str(round(score*100, 2))
            axes[n_row, i % columns].set_title(t1+" : "+t2)
            
    for i in range(rows*columns, rows * columns):
        fig.delaxes(axes[i // columns, i % columns])

    plt.show()