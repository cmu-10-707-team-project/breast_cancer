#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/opt/ASAP/bin')

import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import numpy as np

from skimage import filters, io
from skimage.color import rgb2gray
from tqdm import tqdm_notebook as tqdm

from os import path, remove, mkdir

import pandas as pd


# In[2]:


sys.path.append('../')
from data_process.google_drive_utils import create_drive_service, drive_download_one_file
from data_process.sample_patches import parse_one_annotation_list, sample_one_slide_image

service = create_drive_service()
reader = mir.MultiResolutionImageReader()


# In[3]:


train_df = pd.read_csv('../input/index_train.csv')
test_df = pd.read_csv('../input/index_test.csv')


# In[8]:


train_df.loc[train_df.label == 'tumor'].head()


# In[12]:


input_dir = '../input/train'

def one_work(r):
    try:
        slide_path = path.join(input_dir, r['image_file'])
        mask_path = None
        output_folder = path.join(input_dir, r['id'])

        drive_download_one_file(service, slide_path, r['google_drive_fileid'])

        mkdir(output_folder)

        if (r['label'].lower() == 'tumor'):
            ann_path = path.join(input_dir, r['annotation_file'])
            mask_path = path.join(input_dir, '{}_mask.tif'.format(r['id']))
            parse_one_annotation_list(reader, slide_path, ann_path, mask_path)

        patch_df = sample_one_slide_image(
            reader, r['id'], slide_path, mask_path, output_folder, pos_to_neg_ratio=0.1, neg_sample_rate=0.01, dryrun=False)

        patch_df.to_csv(path.join(output_folder, 'index.csv'), index_label='patch_id')

        remove(slide_path)
        if mask_path is not None:
            remove(mask_path)

        return patch_df
    except:
        return None


# In[ ]:


df = one_work(train_df.iloc[159])


# In[27]:


train_df.shape


# In[6]:


from multiprocessing import Pool

p = Pool(4)
dfs = p.map(one_work, train_df.to_dict(orient='records'))


# In[51]:


df = pd.read_csv('../input/train_backup/tumor_004/index.csv')
df.loc[df['tumor_prob'] > 0]

