#!/usr/bin/python
#     file: train_with_unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse
from datetime import datetime
from multiprocessing import cpu_count

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import *
from keras.models import Model
from keras.optimizers import *

from modeling.data_loader import KerasDataGenerator
from modeling import unet
from modeling.metrics import mean_pred, false_pos_rate, false_neg_rate, \
accuracy, logloss

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=100)

	parser.add_argument('--checkpoint-dir', type=str, default='output')
	parser.add_argument('--tensorboard-dir', type=str, default='data/log')
	parser.add_argument('--model-dir', type=str, default='data/model')
	parser.add_argument('--model-name', type=str, default="unet")
	parser.add_argument('--early-stop-patience', type=int, default=10)

	parser.add_argument('--workers', type=int, default=cpu_count())

	parser.add_argument('--train-index-file-path', type=str, required=True)
	parser.add_argument('--train-input-folder', type=str, required=True)
	parser.add_argument('--val-index-file-path', type=str, required=True)
	parser.add_argument('--val-input-folder', type=str, required=True)

	arg = parser.parse_args()

	########################
	if arg.model_name == 'unet':
	
		train_gen = KerasDataGenerator(
			batch_size =arg.batch_size,
			index_filepath=arg.train_index_file_path,
			input_folder=arg.train_input_folder,
			labeled=True)

		val_gen = KerasDataGenerator(
			batch_size=arg.batch_size,
			index_filepath=arg.val_index_file_path,
			input_folder=arg.val_input_folder,
			labeled=True)

		model = unet.unet()
	
	########################
	elif arg.model_name == 'resnet50':

		train_gen = KerasDataGenerator(
			batch_size =arg.batch_size,
			index_filepath=arg.train_index_file_path,
			input_folder=arg.train_input_folder,
			labeled=True,mask=False)

		val_gen = KerasDataGenerator(
			batch_size=arg.batch_size,
			index_filepath=arg.val_index_file_path,
			input_folder=arg.val_input_folder,
			labeled=True,mask=False)

		from keras.applications.resnet50 import ResNet50
		base_model = ResNet50(weights='imagenet',include_top=False)
		x = GlobalAveragePooling2D()(base_model.output)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(1,activation='sigmoid')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(
	        optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy',
	        metrics=[accuracy, mean_pred, false_pos_rate, false_neg_rate, logloss])
		
	########################
	elif arg.model_name == 'VGG16':

		train_gen = KerasDataGenerator(
			batch_size =arg.batch_size,
			index_filepath=arg.train_index_file_path,
			input_folder=arg.train_input_folder,
			labeled=True,mask=False)

		val_gen = KerasDataGenerator(
			batch_size=arg.batch_size,
			index_filepath=arg.val_index_file_path,
			input_folder=arg.val_input_folder,
			labeled=True,mask=False)

		from keras.applications.vgg16 import VGG16
		base_model = VGG16(weights='imagenet',include_top=False)
		x = GlobalAveragePooling2D()(base_model.output)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(1,activation='sigmoid')(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.compile(
	        optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy',
	        metrics=[accuracy, mean_pred, false_pos_rate, false_neg_rate, logloss])
	########################
	timestamp = datetime.now().strftime('%m-%d-%H%M%S')
	model_path = '{}_{}.hdf5'.format(arg.model_name, timestamp)
	model_checkpoint = ModelCheckpoint(
		model_path, monitor='loss',verbose=1, save_best_only=True)
	tensorboard = TensorBoard(
		log_dir=arg.tensorboard_dir, write_grads=False, write_images=False)
	earlystop = EarlyStopping(
		monitor='val_loss', patience=arg.early_stop_patience)

	model.fit_generator(
		train_gen(), validation_data=val_gen(),
		steps_per_epoch=train_gen.steps_per_epoch, epochs=arg.epochs,
		validation_steps=val_gen.steps_per_epoch,
		use_multiprocessing=True,
		max_queue_size=100,
		workers=arg.workers,
		callbacks=[model_checkpoint, tensorboard, earlystop])
