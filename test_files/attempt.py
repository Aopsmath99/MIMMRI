import torch
import os
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
import numpy as np
import h5py
import fastmri
from matplotlib import pyplot as plt
from skimage import *
from fastmri.data import transforms as T
from fastmri.data import subsample
from fastmri.data.subsample import RandomMaskFunc
from models.simmim import *
from main_finetune import *
#from main_simmim import *
from PIL import Image
import torchvision.transforms as transforms
#import tensorflow as tf
args, configs = parse_option()
logger = create_logger("")
a = build_simmim(configs)

masks = []
filename = '/rfanfs/pnl-zorro/projects/fastmri/data/multicoil_val/file1000000.h5'
hf = h5py.File(filename)
volume_kspace = hf['kspace'][()]
for i in range(20):
    slice_kspace2 = T.to_tensor(volume_kspace[i])
    mask_func = subsample.EquispacedMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8])
    masked_kspace, mask  = T.apply_mask(slice_kspace2, mask_func)
    masks.append(mask)
pathk = '/rfanfs/pnl-zorro/home/kyler/SimMIM/images/kspace/kspacefile1000017/'
pathm = '/rfanfs/pnl-zorro/home/kyler/SimMIM/images/mri/MRIfile1000017/'
yy = (imageio.imread(pathk+'kspace10.png'))
zz = (imageio.imread(pathm+'MRI10.png'))
#plt.imshow(yy, cmap = 'gray')
#plt.show()
#plt.imshow(zz, cmap = 'gray')
#plt.show()
#plt.close()
#print(yy.shape)
#x = np.resize(yy, (4, 192, 192))
#z = np.resize(zz, (4, 192, 192))
#c = np.resize(masks[10], (1, 1, 48, 1))
#fig = plt.figure()
#plt.imshow(x, cmap = 'gray')
#plt.show()
'''
plt.imshow(z, cmap = 'gray')
plt.show()
plt.close()
'''
'''
for i in range(10,20):
    yy = Image.open(pathk+'kspace'+str(i)+'.png')
    zz = Image.open(pathm+'MRI'+str(i)+'.png')
    #yy = (imageio.imread(pathk+'kspace'+str(i)+'.png'))
    #zz = (imageio.imread(pathm+'MRI'+str(i)+'.png'))
    #x = np.resize(yy.T, (4, 192, 192))
    #z = np.resize(zz.T, (4, 192, 192))
    #c = np.resize(masks[i], (1, 1, 48, 1))
    x = yy.resize((192, 192))
    z = zz.resize((192, 192))
    x = np.asarray(x)
    z = np.asarray(z)
    print(x.shape, z.shape)
     
    fig = plt.figure()
    plt.imshow(x, cmap = 'gray')
    plt.show()
    plt.imshow(z, cmap = 'gray')
    plt.show()
    plt.close()
     
    x = np.resize(x, (3, 192, 192))
    z = np.resize(z, (3, 192, 192))
    c = np.resize(masks[i], (1, 1, 48, 1))
    x, c, z = map(torch.tensor, (x, c, z))
    y = a.forward(x, c, z)
'''
def custom_normalize(img):
    img = transform(img)
    mean, std = img.mean([1,2]), img.std([1,2])
    print(mean, std)
    return img
transform = transforms.Compose([
    transforms.ToTensor()
])
yy = Image.open(pathk+'kspace'+str(10)+'.png').convert('RGB')
zz = Image.open(pathm+'MRI'+str(10)+'.png').convert('RGB')
#print(yy)
#yy = (imageio.imread(pathk+'kspace'+str(i)+'.png'))
#zz = (imageio.imread(pathm+'MRI'+str(i)+'.png'))
x = yy.resize((192, 192))
z = zz.resize((192, 192))
x = np.asarray(x)
z = np.asarray(z)
x = np.resize(x, (192, 192, 3))
z = np.resize(z, (192, 192, 3))
print(x.shape, z.shape)
fig = plt.figure()
plt.imshow(x, cmap = 'gray')
plt.show()
plt.imshow(z, cmap = 'gray')
plt.show()
plt.close()
     
x = x.transpose(2, 0, 1)
z = x.transpose(2, 0, 1)
c = np.resize(masks[i], (1, 48, 1))
#x = np.resize(yy, (4, 192, 192))
#z = np.resize(zz, (4, 192, 192))
#c = np.resize(masks[i], (1, 1, 48, 1))
#x = yy.resize((192, 192))
#z = zz.resize((192, 192))
#x = np.asarray(x)
#z = np.asarray(z)
#print(x.shape, z.shape)
#custom_normalize(x)
#c = np.resize(masks[10], (1, 1, 48, 1))
#x = np.resize(yy.T, (192, 192, 4))
#z = np.resize(zz.T, (192, 192, 4))
'''
sampled_image = fastmri.ifft2c(x)
sampled_image_abs = fastmri.complex_abs(sampled_image)
sampled_image_rss = fastmri.rss(sampled_image_abs, dim = 0)
fig = plt.figure()
plt.imshow(sampled_image_rss, cmap = 'gray')
plt.show()

fig = plt.figure()
plt.imshow(x, cmap = 'gray')
plt.show()
plt.imshow(z, cmap = 'gray')
plt.show()
plt.close()
p = []
#[0.5073, 0.5073, 0.5073, 1.0000]
#[0.2534, 0.2534, 0.2534, 0.0000]
'''
p = []
x, c, z = map(torch.tensor, (x, c, z))
#y = a.forward(x, c, z, p)
#ask_toekns = np.resize(mask_tokens, (128, 48, 48))
b, d, f = build_loader(configs, logger, False)
validate(configs, d, a, c, z, p)
#y = main(configs)
#print(p)
#print(min(p))

