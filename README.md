# **Self Supervised Methods Comparison**

## **Bootstrap Your Own Latent (BYOL) - PyTorch**

<img src="static/byol_diagram.png" width="600px"></img>

PyTorch implementation of <a href="https://arxiv.org/abs/2006.07733">BYOL</a> method for self-supervised learning based on <a href="https://cs.stanford.edu/~acoates/stl10/"> STL10 Dataset</a>.

### **Usage**

Set your training params in *hp/BYOL/hp.yml* file. Specifically you can change model backbone and training params (epochs, lr, scheduler, etc.). 

Once your params are ready, run the training script:

```
python train.py --model byol
```
During training, encoder will be saved as pth file.

To extract features from STL10 validation dataset, run the inference script by specifying weights path:
```
python inference.py --model byol --weights checkpoints/byol/byol_2021-12-19-11-35-51/byol_resnet18_epoch_2_loss_0.1236.pth
```
The inference script will save features, labels, and tsne_features in an output folder.

### **Feature Distribution Visualization**

You can visualize TSNE features distribution with <a href="https://streamlit.io">streamlit</a>. To run the visualization webapp:

```
streamlit run visualization/webapp.py
```
You need to select the features folder and then the webapp will display the features distribution.

<img src="static/streamlit_visualization.jpg" width="700px"></img>


### **To-Do List**


[ ] Test Custom Random ColorJitter and GaussianBlur transformations 

[ ] Generalized data loader based on chosen dataset 

[ ] Tensorboard integration

[ ] GradCAM image visualization

[ ] core/train+test as general as possible

[ ] DINO



