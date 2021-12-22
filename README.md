# **Self Supervised Methods Comparison**

## **BYOL - STL10 Dataset**

To change training params, go to *hp/BYOL* folder and change yaml file. 

To run the training script:
```
python train.py --model byol
```
During training, encoder will be saved as pth file.


To run the inference script and saving extracted features with TSNE:
```
python inference.py --model byol
```
During training, encoder will be saved as pth file.

To run visualization webapp with streamlit, run
```
streamlit run visualization/webapp.py
```

## **To-Do List**

[ x ] Inference script

[ x ] Generalized loading dataset

[ x ] Extracted features graphic visualization

[ ] Test Custom Random ColorJitter and GaussianBlur transformations 

[ ] Generalized data loader based on chosen dataset 

[ ] Tensorboard integration

[ ] GradCAM image visualization

[ ] core/train+test as general as possible

[ ] DINO



