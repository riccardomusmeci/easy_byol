import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple
import plotly.express as px

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

    labels = [ categories[idx] for idx in labels ]

    return pd.DataFrame(
        data = {
            "x": features[:, 0],
            "y": features[:, 1],
            "z": features[:, 2],
            "category": labels,
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

def render():

    setup_session()
    st.title("Self Supervised Methods - Features Distribution Visualization")

    folders = os.listdir(OUTPUT_FOLDER)
    option_col, _, visualization_col = st.columns([2, 3, 10])
    option = option_col.selectbox(
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

        with visualization_col:
            st.markdown(f"<h2 style='text-align: center; color: black;'>{dataset.upper()} Features</h2>", unsafe_allow_html=True)
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

            st.write(fig)
        
render()
