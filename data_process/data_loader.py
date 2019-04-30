import os

from os import path, cpu_count

import pandas as pd

import numpy as np

import tensorflow as tf

from PIL import Image

PATCH_SIZE = 256
N_CHANN = 3


class TumorPathGenerator:
    def __init__(self, index_filepath, input_folder, is_train):
        self.input_folder = input_folder

        index_df = pd.read_csv(index_filepath)

        if is_train:
            # re-balance
            tumor_index = index_df.loc[index_df['tumor_prob'] > 0]
            normal_index = index_df.loc[index_df['tumor_prob'] == 0]

            print(tumor_index.shape)
            print(normal_index.shape)

            # negative sampling
            sampled_normal = normal_index.sample(
                n=tumor_index.shape[0], replace=True)

            index_df = pd.concat([tumor_index, sampled_normal], axis=0)

        self.index_df = index_df

    def __call__(self):
        for _, r in self.index_df.iterrows():
            patch_path = path.join(
                self.input_folder, r['slide_id'], r['filename'])
            yield patch_path

class KerasDataGenerator:
    def __init__(self, index_filepath, input_folder, is_train,batch_size):
        self.input_folder = input_folder
        self.batch_size = batch_size
        index_df = pd.read_csv(index_filepath)

        if is_train:
            # re-balance
            tumor_index = index_df.loc[index_df['tumor_prob'] > 0]
            normal_index = index_df.loc[index_df['tumor_prob'] == 0]

            # negative sampling
            sampled_normal = normal_index.sample(
                n=tumor_index.shape[0], replace=True)

            index_df = pd.concat([tumor_index, sampled_normal], axis=0)

        self.index_df = index_df
        self.num_data = index_df.shape[0]
    
    def __call__(self):
       
        batch_data = []
        batch_label = []
        for idx, r in self.index_df.iterrows():

            patch_path = path.join(
                self.input_folder, r['slide_id'], r['filename'])

            image = Image.open(patch_path)
            np_img = np.asarray(image)
            batch_data.append(np_img[:, :, 0:-1])
            label = np_img[:, :, -1]
            label = label[..., np.newaxis]
            batch_label.append(label)
          
            if len(batch_data) == self.batch_size or idx == (self.num_data-1):
                yield np.asarray(batch_data), np.asarray(batch_label)
                batch_data = []
                batch_label = []

class KerasTestDataGenerator:
    def __init__(self,test_folder):
        self.test_folder = test_folder

    def __call__(self):
        for filename in os.listdir(self.test_folder):
            if filename.endswith(".png"): 
                patch_path = self.test_folder + filename
                image = Image.open(patch_path)
                np_img = np.asarray(image)[:, :, 0:-1] # onyl for using training as test,need to remove
                np_img = np.expand_dims(np_img, axis=0)
                yield np_img


class TumorPatchDatasetInputFun:
    def __init__(self, batch_size, shuffle_buffer_size, *args, **kwargs):
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

        self.gen = TumorPathGenerator(*args, **kwargs)
        self.dataset = tf.data.Dataset\
            .from_generator(self.gen, tf.string)\
            .map(self.load_and_preprocess_patch,
                 num_parallel_calls=cpu_count())\
            .map(self.patch_augmentation, num_parallel_calls=cpu_count())\
            .map(self.gen_labeled_data, num_parallel_calls=cpu_count())

        if shuffle_buffer_size is not None:
            self.dataset = self.dataset.shuffle(shuffle_buffer_size)

        self.dataset = self.dataset.batch(batch_size)

    def __call__(self, *args, **kwargs):
        return self.dataset.make_one_shot_iterator()

    def load_and_preprocess_patch(self, image_path):
        patch_file = tf.read_file(image_path)
        patch = tf.image.decode_png(patch_file, channels=4)
        patch = tf.cast(patch, tf.float32)
        patch = tf.div(patch, 255.)
        return patch

    def gen_labeled_data(self, patch):
        patch.set_shape([PATCH_SIZE, PATCH_SIZE, N_CHANN + 1])
        return patch[:, :, 0:-1], patch[:, :, -1]

    def patch_augmentation(self, patch):
        angle = np.random.choice([0, np.pi / 2, np.pi, 1.5 * np.pi])
        return tf.contrib.image.rotate(patch, angle)
