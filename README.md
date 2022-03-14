# **easy_byol**

## **Bootstrap Your Own Latent (BYOL) - PyTorch**

<p align="center">
    <img src="static/byol_diagram.png" width="600px"></img>
</p>

PyTorch custom implementation of <a href="https://arxiv.org/abs/2006.07733">BYOL</a>, a self-supervised learning method, for <a href="https://cs.stanford.edu/~acoates/stl10/"> STL10</a>, <a href="https://www.kaggle.com/michaelfumery/unlabeled-stanford-dags-dataset"> Dogs</a>, or <a href="https://www.cs.toronto.edu/~kriz/cifar.html"> CIFAR10 </a> datasets.

## **Dataset**

### **Dogs Dataset**

* Download dataset from kaggle (<a href="https://www.kaggle.com/michaelfumery/unlabeled-stanford-dags-dataset"> Dogs Dataset </a>)
* Create a *data/dogs* folder structure in the project directory
* Put train, test, and list_breeds.csv into data/dogs directory
* Resize all dataset images in a square format (e.g. 96x96) to speed up the training process

You can use the following code in a custom script:
```
import os
from tqdm import tqdm
from PIL import Image
from glob2 import glob

data_path = "data/dogs"
files = [f for f in glob(os.path.join(data_path, "*", "*.jpg"))]
for fpath in tqdm(files, total=len(files)):
    im = Image.open(fpath).resize(size=(96, 96))
    im.save(fpath)
```


## **BYOL Train + Inference**

Set your training params in *config/BYOL/config.yml* file. You can change dataset (STL10/dogs/CIFAR10), model backbone and training params (epochs, lr, scheduler, etc.). 

Once your params are ready, run the training script:

```
python train.py --model byol
```
During training, the code will create an output folder within the checkpoints folder, structured as follows:

```
project
│   README.md
│   
└─── checkpoints
│   │         └── byol
│   │                │
│   │                └── byol_DATE
│   │                             │ weights/*.pth
│   │                             │ config.yml

```
Within the weights folder, only the encoder pth will be saved.

To extract features from your validation dataset, run the byol inference script by specifying weights path.:
```
python inference.py --model byol --weights checkpoints/byol/byol_2021-12-19-11-35-51/byol_resnet18_epoch_203_loss_0.1236.pth --config checkpoints/byol/byol_2021-12-19-11-35-51/config.yml
```

The inference script will save features, labels, and tsne_features in an output folder.

### **WebApp Visualization**

Before running the visualization webapp, you must generate the images for the STL10 and the CIFAR10 dataset. To do so, run the scripts in *support* folder (*stl10.py* and *cifar10.py*). They will generate images of the test dataset into *data* folder.


You can visualize TSNE features distribution as well as some samples with <a href="https://streamlit.io">streamlit</a>. To run the visualization webapp:

```
streamlit run frontend/webapp.py
```
You need to select the features folder and then the webapp will display the features distribution. You can also interact with the 3D graph (e.g. zooming in/out, selecting categories to show, etc.). 

By clicking on "View Random Samples" button the webapp will show a sample and the closest 5 images based on the features extracted by the model.

<p align="center">
    <img src="static/streamlit.jpg" width="700px"></img>
</p>

### **GradCAM Visualization**
The repository provides *show_gradcam.ipynb* to show GradCAM visualization from BYOL model.

You can choose the model weights as well as the image to process.

## **Classifier Train + Inference**
You can train a classifier after a BYOL based backbone has been trained. Also, you can specify configuration parameters at *config/classifier/config.yml*. 

To do so, please use the *train.py* script by specifying the BYOL checkpoints dir (e.g. checkpoints/byol/[BYOL_TRAINING_OUTPUT_FOLDER]) and the weights within the *weights* folder:

```
python train.py --model byol --ssl-dir checkpoints/byol/byol_2022-03-13-12-51-30 --ssl-pth byol_resnet18_epoch_203_loss_0.0323.pth
```

Finally, you can check your model accuracy on validation dataset using the *inference.py* script:

```
python inference_byol.py --model classifier --weights checkpoints/classifier/classifier_2022-03-13-13-18-42/classifier_resnet18_epoch_73_loss_2.6207_acc_87.5623 --config checkpoints/byol/classifier_2022-03-13-13-18-42/config.yml
```

## **Notes**
The code is intended to deliver and easier and modular access to the BYOL method. You should be able to easy extend the code with few steps. 

### **Extension Example - Loss**
If you want to add another loss support to the repository you must:
* define the loss criterion within the *src/loss/* folder
* in *src/loss/loss.py*, import your loss and add an if condition in the function that, if verified, returns your new loss
 ```
from src.loss.new_loss import NewLoss

def get_loss_fn(loss: str = "norm_mse"):
    ...
    if loss == "new_loss":
        return NewLoss()
```
* in the config yml file (wither byol or classifier), modify the field regarding the loss, for example:
```
loss:
  type: new_loss
```
You can extend the code as shown above for almost every configuration in the config.yml of both classifier and byol model. 

**It is intended that you might need some code refactoring if you start adding many different excting stuff :)**


## **To-Do**
- [ ] Image Augmentations porting to Albumentations
- [ ] Support for other backbone models
