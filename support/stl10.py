import os
import numpy as np
from tqdm import tqdm
from imageio import imsave

def read_all_images(path: str) -> np.array:
    """returns all dataset images

    Args:
        path (str): path to STL10 bin file with images

    Returns:
        np.array: all images
    """

    with open(path, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def save_image(image: np.array, filename: str):
    """saves an image

    Args:
        image (np.array): image
        filename (str): image filename
    """
    imsave("%s.png" % filename, image, format="png")

def save_images(images: np.array, labels: np.array, dst_dir: str, mode: str):
    """saves images in the form idx_labelid.png

    Args:
        images (np.array): images
        labels (np.array): labels
        dst_dir (str): where to save images
        mode (str): test/train
    """
    i = 0
    dst_dir = dst_dir + '/' + mode + '/'
    print(f"Saving data into {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    for image in tqdm(images, position=0):
        label = labels[i] 
        filename = os.path.join(dst_dir, f"{i}_{label}")
        save_image(image, filename)
        i = i+1

def read_labels(path) -> np.array:
    """read labels numpy array from a given path

    Args:
        path (str): path to labels saved as bin file

    Returns:
        np.array: labels array
    """

    with open(path, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# target directory + original test directory
data_dir = "../data/stl10_test"
test_dir = "../data/stl10_binary"
# reading labels for test images
test_labels = read_labels(
    path = os.path.join(test_dir, "test_y.bin")
)
# reading all images
test_images = read_all_images(
    path = os.path.join(test_dir, "test_X.bin")
)

# saving images in the form (idx_labelid.png) in test folder
save_images(
    images=test_images, 
    labels=test_labels,
    dst_dir=data_dir,
    mode="test"
)




