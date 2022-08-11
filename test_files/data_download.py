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
from fastmri.data.subsample import RandomMaskFunc
from models.simmim import *
from main_finetune import *
def kspace(data, coils, path, counter, cmap=None):
    fig = plt.figure()
    #for i in range(coils)
    #plt.subplot(1, coils, i+1)
    data2 = T.to_tensor(data)
    img_rss = fastmri.rss(data2, dim=0)
    #print(img_rss.shape)
    img_rss = img_rss.transpose(2, 0)
    img_rss = img_rss.transpose(2, 1)
    print(img_rss.shape)
    transform = Tr.ToPILImage()
    img_rss = transform(img_rss)
    img_rss.show()
    plt.axis("off")
    #plt.savefig(path+'kspace'+str(i)+'.png', bbox_inches = 'tight')
    plt.close()
    print("kspace"+str(i)+" saved to " + path)
    bruh
    #img = np.asarray(fig)
    #imageio.imwrite(path+'kspace'+str(i)+'.png', img)
def MRI(img, coils, path, counter, cmap = None):
    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.axis("off")
    plt.show()
    #plt.savefig(path+'MRI'+str(i)+'.png', bbox_inches = 'tight')
    print("MRI" + str(i) + " saved to " + path)
    plt.close()
    bruhlol
if __name__ == '__main__':
    l = [1001955, 1001959, 1001968, 1001977, 1001983, 1001984, 1001995, 1001997, 1002002, 1002007, 1002021, 1002035, 1002067, 1002145, 1002155, 1002159, 1002187, 1002214, 1002252, 1002257, 1002274, 1002280, 1002340, 1002351, 1002377, 1002380, 1002382, 1002389, 1002404, 1002412, 1002417, 1002436, 1002451, 1002515, 1002526, 1002538, 1002546, 1002570]
    #k = h5py.File('imagesh5.h5', "w")
    #m = h5py.File('/rfanfs/pnl-zorro/projects/fastmri/imagesh5/mri', "w")
    for j in l:
        filename = '/rfanfs/pnl-zorro/projects/fastmri/data/multicoil_val/file' + str(j)+'.h5'
        hf = h5py.File(filename)
        volume_kspace = hf['kspace'][()]
        volume_MRI = hf['reconstruction_rss'][()]
        #print(volume_MRI.shape)
        number = str(j)
        pathm = '/rfanfs/pnl-zorro/projects/fastmri/kyler/mri/MRIfile'
        pathk = '/rfanfs/pnl-zorro/projects/fastmri/images/test/kspacefile'
        pathm = pathm+number+'/'
        pathk = pathk+number+'/'
        #ks = k.create_dataset('kspacefile' + str(j), (volume_kspace.shape[0], volume_kspace.shape[2], volume_kspace.shape[3]), dtype='i')
        #mr = k.create_dataset('mri' + str(j), (volume_kspace.shape[0], volume_MRI.shape[1], volume_MRI.shape[2]), dtype='i')
        #print('saving file ' +str(j) + ' now')

        #os.mkdir(pathm)
        #os.mkdir(pathk)
        for i in range(volume_kspace.shape[0]):
            #dset = ks.create_dataset("kspace" + str(i), data=np.log(np.abs(volume_kspace[i]+1e-9)), dtype='i')
            #data2 = np.log(np.abs(volume_kspace[i]+1e-9))
            data2 = volume_kspace[i]
            #ks[i] = img
            #print('saved kspace' + str(i) + ' as hdf5 file')
            #mr[i] = volume_MRI[i]
            #dset2 = ks.create_dataset("mri" + str(i), data=volume_MRI[i], dtype='i')
            #print('saved mri' + str(i) + ' as hdf5 file')
            #try:

            kspace(volume_kspace[i], 15, pathk, i, cmap = 'gray')
            MRI(volume_MRI[i], 15, pathm, i, cmap = 'gray') 
            '''
            except:
                print("all images read, moving to next")
                break
            '''
