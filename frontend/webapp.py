import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple
import plotly.express as px
from glob2 import glob
from PIL import Image
from scipy.spatial.distance import cdist as dist

OUTPUT_FOLDER = "output"
CHECKPOINT_FOLDER = "checkpoints/byol"
st.set_page_config(layout="wide")


def load_numpy_data(folder: str, tsne: bool = True) -> Tuple[np.array, np.array]:
    """loads numpy data for features and labels

    Args:
        folder (str): folder with numpy file (features.npy, labels.npy)
        tsne (bool): whether to load tsne features. Defaults to True.

    Returns:
        Tuple[np.array, np.array]: features, labels
    """
    features_f = "tsne_features.npy" if tsne else "features.npy"
    features = np.load(file=os.path.join(folder, features_f))
    labels = np.load(file=os.path.join(folder, "labels.npy"))

    return features, labels

def get_colormap(dataset: str) -> dict:
    """returns color map for dataset categories

    Args:
        dataset (str): dataset

    Returns:
        dict: color map
    """
    if dataset=="STL10":
        return {
            "airplane": "blue", 
            "bird": "green", 
            "car": "red", 
            "cat": "gray", 
            "deer": "black", 
            "dog": "orange",
            "horse": "lime",
            "monkey": "gold",
            "ship": "brown",
            "truck": "pink"
        }

    if dataset == "dogs":
        return {
            "Chihuahua": "blue",
            "Shih": "green",
            "beagle": "red",
            "Staffordshire_bullterrier": "gray",
            "Yorkshire_terrier": "black",
            "Australian_terrier": "orange",
            "golden_retriever": "lime",
            "Labrador_retriever": "gold",
            "English_setter": "brown",  
            "cocker_spaniel": "pink",
            "Border_collie": "turquoise",
            "Rottweiler": "bisque",
            "German_shepherd": "silver",
            "French_bulldog": "lightcoral",
            "Siberian_husky": "mistyrose"
        }

def load_categories(dataset: str) -> list:
    """loads categories for a dataset
    Args:
        dataset (str): dataset name. Defaults to "data/stl10_binary/class_names.txt".

    Returns:
        list: list of categories
    """
    if dataset == "STL10":
        with open("data/stl10_binary/class_names.txt", "r") as f:
            categories = f.read().split("\n")
        f.close()

    if dataset == "dogs":
        df = pd.read_csv("data/dogs/list_breeds.csv", sep=";")
        return list(df["Breed"])

    if dataset == "CIFAR10":
        categories = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return categories

def get_df(dataset: str, features: np.array, labels: np.array) -> pd.DataFrame:
    """Computes PCA on features

    Args:
        features (np.array): features
        labels (np.array): labels

    Returns:
        pd.DataFrame: DataFrame with X, Y, Z and Target cols
    """
    print(f"Getting data for dataset {dataset}")
    categories = load_categories(dataset)

    labels = [ categories[idx].replace("_", " ").lower() for idx in labels ]

    return pd.DataFrame(
        data = {
            "x": features[:, 0],
            "y": features[:, 1],
            "z": features[:, 2],
            "category": labels
        }
    )

def get_dataset(training_info_txt: str) -> str:
    """get dataset name from training_info.txt

    Args:
        training_info_txt (str): path to training_info.txt in checkpoints dir

    Returns:
        str: dataset name
    """
    with open(training_info_txt, "r") as f:
        dataset = f.read().split("\n")[0].split(":")[-1].replace(" ", "")
        return dataset

def setup_session():
    if 'folder' not in st.session_state:
        st.session_state.folder = "None"

def get_idxs(features: np.array, n: int = 5) -> list:
    """Gets random samples to show (one anchor and closest samples)

    Args:
        features (np.array): features
        n (int, optional): how many samples to show

    Returns:
        list: list of indices
    """
    index = np.random.choice(features.shape[0]) 
    rand_sample = features[index]
    dist_mat = dist([rand_sample], features, metric="cosine")
    idxs = dist_mat.argsort()[0][:n+1]
    return idxs

def get_labels(dataset: str, idxs: str) -> list:
    """Get labels for given images beloging to a specific dataset

    Args:
        dataset (str): dataset name
        idxs (str): subset of images

    Returns:
        list: labels
    """
    if dataset == "dogs":
        labels_df = pd.read_csv("data/dogs/list_breeds.csv", sep=";")
        id2name = {
            row["Id"]: row["Breed"].replace("_", " ").capitalize() for _, row in labels_df.iterrows()
        }
        imgs = [ f.split(os.sep)[-1] for f in glob("data/dogs/test/*.jpg") ]
        labels = []
        for idx in idxs:
            _id = imgs[idx].split("_")[0]
            labels.append(id2name[_id])
        return labels

    if dataset == "STL10":
        with open("data/stl10_binary/class_names.txt", "r") as f:
            class_names = f.read().split("\n")
        id2name = {
            str(i+1): class_names[i] for i in range(len(class_names))
        }
        imgs = [ f.split(os.sep)[-1] for f in glob("data/stl10_test/test/*.png")]
        labels = []
        for idx in idxs:
            _id = imgs[idx].split(".")[0].split("_")[-1]
            labels.append(id2name[_id])
        return labels

    if dataset == "CIFAR10":
        categories = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        imgs = [ f.split(os.sep)[-1] for f in glob("data/cifar10_test/test/*.jpg") ] 
        labels = []
        for idx in idxs:
            cat_idx = int(imgs[idx].split(".")[0].split("_")[-1])
            labels.append(categories[cat_idx])

        return labels

def get_imgs_to_show(dataset: str, features: np.array):
    """Gets random samples to show (anchor + closest samples)

    Args:
        dataset (str): dataset name
        features (np.array): features

    Returns:
        Tuple: anchor, closest_samples_img, labels
    """
    if dataset == "dogs":
        data_path = "data/dogs/test/*.jpg"
    if dataset == "STL10":
        data_path = "data/stl10_test/test/*.png"
    if dataset == "CIFAR10":
        data_path = "data/cifar10_test/test/*.jpg"

    img_paths = [f for f in glob(data_path)]
    idxs = get_idxs(features=features)
    ### Random Sample 
    anchor_img = Image.open(img_paths[idxs[0]])
    ### Concatenating Closest samples
    img_size = (128, 128) if dataset != "CIFAR10" else (64, 64)
    images = [Image.open(img_paths[idx]).resize(img_size) for idx in idxs[1:]]
    widths, heights = zip(*(i.size for i in images))

    total_width, max_height= sum(widths), max(heights)
    closest_samples_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        closest_samples_img.paste(im, (x_offset,0))
        x_offset += im.size[0]
    
    return anchor_img, closest_samples_img, get_labels(dataset=dataset, idxs=idxs)

def render():

    setup_session()
    st.title("Self Supervised Methods")

    folders = os.listdir(OUTPUT_FOLDER)
    features_col, _, images_col = st.columns([6, 3, 4])
    option = features_col.selectbox(
        'Which output folder you want to analyze?',
        ["None"] + folders, 
    )

    if option != "None":
        st.session_state.folder = option
        dataset = get_dataset(training_info_txt=os.path.join(CHECKPOINT_FOLDER, option, "training_info.txt"))
        folder = f"{OUTPUT_FOLDER}/{option}"
        features, labels = load_numpy_data(
            folder=folder,
            tsne=True
        )

        with features_col:
            _dataset = "Unsupervised Dogs" if dataset == "dogs" else dataset
            features_col.markdown(f"<h3 style='text-align: center; color: black;'>{_dataset} Features</h3>", unsafe_allow_html=True)
            feat_distr = get_df(
                dataset=dataset,
                features=features,
                labels=labels
            )
            fig = px.scatter_3d(
                feat_distr, 
                x="x", 
                y="y", 
                z="z", 
                color="category",
                color_discrete_map = get_colormap(dataset=dataset)
            )

            features_col.write(fig)
        images_col.markdown("Click on this button to view some samples",  unsafe_allow_html=True)
        if images_col.button('View Random Samples'):
            images_col.markdown(f"<h3 style='text-align: center; color: black;'>Random Sample - Closest Images</h3>", unsafe_allow_html=True)
            
            anchor, closest_samples, labels = get_imgs_to_show(dataset=dataset, features=features)
            images_col.markdown(f"<h4 style='text-align: left; color: black;'><br>Random Sample ({labels[0]})</h4>", unsafe_allow_html=True)
            anchor_width = 128 if dataset != "CIFAR10" else 64
            images_col.image(anchor, width=anchor_width)

            ### Concatenating Closest samples
            labels_str = " - ".join(labels[1:])
            images_col.markdown(f"<h4 style='text-align: left; color: black;'><br>Closest Samples ({labels_str})</h4>", unsafe_allow_html=True)
            images_col.image(closest_samples, width=closest_samples.size[0])
                

render()