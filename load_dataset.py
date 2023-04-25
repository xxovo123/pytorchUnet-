import numpy as np
import os
import cv2 
# path_test = "test_npz_seg_512x512_s1"
# path_train ='train_npz_seg_512x512_s1'

def load_data(path):
    data = []
    mask = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            imdata = cv2.imread(os.path.join(path,filename))
            maskfilename = filename.replace('.png', '_target.npz')
            maskdata = np.load(os.path.join(path,maskfilename), allow_pickle=True)['arr_0']
            data.append(imdata)
            mask.append(maskdata)
    data = np.array(data)
    mask = np.array(mask)
    return data,mask

