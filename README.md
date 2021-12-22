# **Self Supervised Methods Comparison**

## **BYOL**
BYOL on STL10 Dataset. 

To change training params, go to *hp/BYOL* folder and change yaml file. 

To run the training script:
```
python train.py --model byol
```
During training, encoder will be saved as pth file.


To run the inference script and saving extracted features:
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

[ ] Extracted features graphic visualization

[ ] Test Custom Random ColorJitter and GaussianBlur transformations 

[ ] Generalized data loader based on chosen dataset 

[ ] Tensorboard integration

[ ] GradCAM image visualization

[ ] core/train+test as general as possible

[ ] DINO



