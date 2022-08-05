# MIMMRI
Welcome to Masked Image Modeling MRI Reconstruction Imaging (MIMMRI)! This respository contain code that evaluates and modifies basline code for masked image modeling. Specifically, this code was used for the purpose of evaluating masked image modeling for MRI reconstruction. To get started with this repository, follow the below instructions.

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
torchrun simmim_main.py --cfg=swin100epwin6 --amp-opt-level='O0' --local_rank=1
```
Config.py contain several other valuable configs, and other encoders can be found in the models folder along with the main SimMIM folder. Each of these encoders were modified by me for the purpose of reconstruction (and formatting for the SimMIM baseline model), but improvements are always needed and greatly welcome. 

This repository contains dozens of useful implementations that are not all used in the above code. Feel free to use any implementation found in this repository for any use. 

### Acknowledgements:
I would like to thank Arghya Pal, PhD and Kevin Cho, PhD from the Psychiatry Neuroimaging lab at the Harvard Medical School for helping me setup up this repository and project, and for answering the multitude of questions I asked while I worked. 

### Paper
Stay tuned!
