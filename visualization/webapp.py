import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple
import plotly.express as px

OUTPUT_FOLDER = "output"
st.set_page_config(layout="wide")


def setup_session():
    if 'folder' not in st.session_state:
        st.session_state.folder = "None"

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

def STL10_colormap() -> dict:
    """Defines color map for STL10 dataset

    Returns:
        dict: color map
    """
    color_map = {
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

    return color_map

def load_STL10_categories(path: str = "data/stl10_binary/class_names.txt") -> list:
    """loads categories for STL10 dataset

    Args:
        path (str, optional): path to txt file with categories names. Defaults to "data/stl10_binary/class_names.txt".

    Returns:
        list: list of categories
    """

    with open(path, "r") as f:
        categories = f.read().split("\n")
    f.close()

    return categories

def get_df(features: np.array, labels: np.array) -> pd.DataFrame:
    """Computes PCA on features

    Args:
        features (np.array): features
        labels (np.array): labels

    Returns:
        pd.DataFrame: DataFrame with X, Y, Z and Target cols
    """

    categories = load_STL10_categories()

    labels = [ categories[idx] for idx in labels ]

    return pd.DataFrame(
        data = {
            "x": features[:, 0],
            "y": features[:, 1],
            "z": features[:, 2],
            "category": labels,
        }
    )

def render():

    setup_session()
    st.title("Self Supervised Methods - Features Distribution Visualization")

    folders = os.listdir(OUTPUT_FOLDER)
    option_col, _, visualization_col= st.columns([2, 3, 10])
    option = option_col.selectbox(
        'Which output folder you want to analyze?',
        ["None"] + folders, 
    )

    if option != "None":
        st.session_state.folder = option
        folder = f"{OUTPUT_FOLDER}/{option}"
        features, labels = load_numpy_data(
            folder=folder,
            tsne=True
        )

        with visualization_col:
            st.markdown("<h2 style='text-align: center; color: black;'>TSNE Features</h2>", unsafe_allow_html=True)
            feat_distr = get_df(
                features=features,
                labels=labels
            )
            fig = px.scatter_3d(
                feat_distr, 
                x="x", 
                y="y", 
                z="z", 
                color="category",
                color_discrete_map = STL10_colormap()
            )

            st.write(fig)
        
render()
