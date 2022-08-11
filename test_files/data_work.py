import torch
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
#import tensorflow as tf
class MyDataset(Dataset):
    def __init__(self, values):
        super(MyDataset, self).__init__()

        self.values = values
    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]
    def __iter__(self):
        return iter(Dataset)


def main(dataloader):
    for i, batch in enumerate(dataloader):
        print(i,'and',batch)

def kspace(data, coils, path, counter, cmap=None):
    fig = plt.figure()
    #for i in range(coils)
    #plt.subplot(1, coils, i+1)
    data2 = T.to_tensor(data)
    img_rss = fastmri.rss(data2, dim=0)
    plt.imshow(np.abs(img_rss), cmap = cmap)
    #plt.savefig(path+'kspace'+str(i)+'.png')
    plt.close()
    print("kspace"+str(i)+"saved")
    #img = np.asarray(fig)
    #imageio.imwrite(path+'kspace'+str(i)+'.png', img)
def MRI(img, coils, path, counter, cmap = None):
    
    fig = plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.savefig(path+'MRI'+str(i)+'.png')
    print("MRI" + str(i) + "saved")
    plt.close()
    '''
    slice_kspace2 = T.to_tensor(img[8])
    slice_image = fastmri.ifft2c(slice_kspace2)
    slice_image_abs = fastmri.complex_abs(slice_image)
    slice_images_rss = fastmri.rss(slice_image_abs, dim = 0)
    plt.imshow(np.abs(slice_image_rss.numpy()), cmap = 'gray')
    plt.show()
    '''
if __name__ == '__main__':
    filename = '/rfanfs/pnl-zorro/projects/fastmri/multicoil_val/file1000182.h5'
    hf = h5py.File(filename)
'''
    #print('Keys:', list(hf.keys()))
    #print('Attrs:', dict(hf.attrs))
    volume_kspace = hf['kspace'][()]
    volume_MRI = hf['reconstruction_rss'][()]
    #print(volume_MRI.shape)
    #pathm = '/rfanfs/pnl-zorro/home/kyler/SimMIM/MRIfile1000182/'
    #pathk = '/rfanfs/pnl-zorro/home/kyler/SimMIM/kspacefile1000182/'
    #for i in range(38):
        #kspace(np.log(np.abs(volume_kspace[i])+1e-9), 15, pathk, i, cmap = 'gray')
        #MRI(volume_MRI[10], 15, pathm, i, cmap = 'gray'
    
    #pathk = '/rfanfs/pnl-zorro/home/kyler/SimMIM/kspacefile1000182/'
    #pathm = '/rfanfs/pnl-zorro/home/kyler/SimMIM/MRIfile1000182/'
    y=np.array(imageio.imread(pathk+'kspace0'+'.png'))
    for j in range(1, 38):
        yy = (imageio.imread(pathk+'kspace'+str(j)+'.png'))
        y=np.append(y, yy)
    z=np.array(imageio.imread(pathm+'MRI0'+'.png'))
    for j in range(1, 38):
        yy = (imageio.imread(pathm+'MRI'+str(j)+'.png'))
        z=np.append(y, yy)

k = []
k.append([y])
k.append([z])
'''
k = [1]
dataset = MyDataset(k)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
'''
masks = []
volume_kspace = hf['kspace'][()]
for i in range(38):
    for j in range(15):
        slice_kspace2 = T.to_tensor(volume_kspace[j])
        mask_func = RandomMaskFunc(center_fractions=[0.4], accelerations=[8])
        masked_kspace, mask  = T.apply_mask(slice_kspace2, mask_func)
        masks.append(mask)
'''
args, configs = parse_option()
logger = create_logger("")
a = build_simmim(configs)
b, d, f = build_loader(configs, logger, False)
validate(configs, d, a)
'''
    mask = masks[i]
    pathk = '/rfanfs/pnl-zorro/home/kyler/SimMIM/kspace_file1000182/'
    yy = imageio.imread(path+'kspace'+str(i)+'.png')
    x = tf.image.resize(yy, (4, 192, 192))
    print(x.shape)
    print(a.forward(x, mask))
'''


 
