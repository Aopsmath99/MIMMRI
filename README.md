# MIMMRI
Welcome to Masked Image Modeling MRI Reconstruction Imaging (MIMMRI)! This respository contain code that evaluates and modifies basline code for masked image modeling. Specifically, this code was used for the purpose of evaluating masked image modeling for MRI reconstruction. To get started with this repository, follow the below instructions.

## Introduction
This study follows the use of Masked Image Modeling (MIM). It primarily uses code from the baseline SimMIM model (https://github.com/microsoft/SimMIM), which has been modified for reconstruction purposes. The model uses two primary encoders: The Swin Transformer and the Vision Transformer. Some changes were made to the overall algorithm of these models, consisting largely of removing extraneous methods and adding additional masking functions to simulate undersampling. The primary changes made were to hyperparameters and implementation, as well as the addition of new encoders like the SwinRecNet. It is important to note that, so far, this model only produces a reconstructed k-space, and the model is currently being worked on to produce the actual MRI image (still working out specifics). The basic Inverse Fourier transform is not applicable in this case since the image loses its complex dimension when save as a PNG file.

## Dataset
This study makes use of Facebook's fastmri dataset (https://fastmri.org/). Specifically, this study uses only the validation set for knee-MRI images, split 80/20 for train/test. The data is downloaded as an h5 file, and each h5 file contains approximately 35* images of k-space and MR images *(30-40 per file). This data was saved as a .png files before being passed thorugh the model, and augmented only by resizing (mostly centercrop) to keep details. 

## Setup
Run the following commands to obtain the necessary setup requirements:

```
#clone apex repo
git clone https://github.com/NVIDIA/apex
cd apex

#install apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

#clone repo
git clone https://github.com/Aopsmath99/MIMMRI
cd MIMMRI

#install requirements
pip install -r requirements.txt
```

The model that found the most success can be trained and run below:

## Train given model and evaluate
```
torchrun simmim_main.py --cfg=swin100epwin6.yaml --amp-opt-level='O0' --local_rank=1
```
Config.py contain several other valuable configs, and other encoders can be found in the models folder along with the main SimMIM folder. Each of these encoders were modified by me for the purpose of reconstruction (and formatting for the SimMIM baseline model), but improvements are always needed and greatly welcome. This project was conducted over a brief 6-week time period, meaning there is much room for improvement. This repo will likely continue to be updated thorughout the next year with improvements.

This repository contains dozens of useful implementations that are not all used in the above code. Feel free to use any implementation found in this repository for any use. 

### Results
In current state run of the above command, the model achieved structural similarity values of over 99.5% and loss values of less than 0.01 during validation, with training loss values dipping below 0.005. The full dataset for multiple models will be made available shortly.

### Acknowledgements:
I would like to thank Arghya Pal, PhD and Kevin Cho, PhD from the Psychiatry Neuroimaging lab at the Harvard Medical School for helping me setup up this repository and project, and for answering the multitude of questions I asked while I worked. 

### Paper
Coming Soon
