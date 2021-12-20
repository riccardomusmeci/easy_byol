# **Self Supervised Methods Comparison**

## **BYOL**
BYOL on STL10 Dataset. 

To change training params, go to *hp/BYOL* folder and change yaml file. 

To run the training script:
```
python train.py --model byol
```
During training, encoder will be saved as pth file.



## **To-Do List**

[ x ] Inference script

[ x ] Generalized loading dataset

[ ] Generalized data loader based on chosen dataset
 
[ ] Extracted features graphic visualization

[ ] ColorJitter and GaussianBlur transformations as nn.Module supporting randomness

[ ] Tensorboard integration

[ ] GradCAM image visualization

[ ] core/train+test as general as possible

[ ] DINO



