#!/usr/bin/python
#     file: train.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count
from os import path

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from modeling.data_loader import KerasDataGenerator
from modeling.model_factory import get_model

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
	parser.add_argument('--log-freq', type=int, default=10)
	parser.add_argument('--model-dir', type=str, default='data/model')
	parser.add_argument('--model-name', type=str, default="unet")
	parser.add_argument('--model-suffix', type=str, default='')
	parser.add_argument('--early-stop-patience', type=int, default=3)

	parser.add_argument('--workers', type=int, default=cpu_count())

	parser.add_argument('--train-index-file-path', type=str, required=True)
	parser.add_argument('--train-input-folder', type=str, required=True)
	parser.add_argument('--val-index-file-path', type=str, required=True)
	parser.add_argument('--val-input-folder', type=str, required=True)

	# learning rate
	parser.add_argument('--lr', type=float, default=1e-3)
	# weight 
	parser.add_argument('--weights', type=str, default=None)
	
	arg = parser.parse_args()

	label_mask = arg.model_name == 'unet'

	train_gen = KerasDataGenerator(
		batch_size=arg.batch_size,
		index_filepath=arg.train_index_file_path,
		input_folder=arg.train_input_folder,
		labeled=True, mask=label_mask)

	val_gen = KerasDataGenerator(
		batch_size=arg.batch_size,
		index_filepath=arg.val_index_file_path,
		input_folder=arg.val_input_folder,
		labeled=True, mask=label_mask, eval=True)

	########################
	model = get_model(**arg.__dict__)
	########################

	timestamp = datetime.now().strftime('%m-%d-%H%M%S')
	saved_model_name = '{}_{}_{}'.format(
		arg.model_name, arg.model_suffix, timestamp)
	model_checkpoint = ModelCheckpoint(
		path.join(arg.model_dir, saved_model_name + '.hdf5'), monitor='loss', verbose=1, save_best_only=True)
	tensorboard = TensorBoard(
		log_dir=path.join(arg.tensorboard_dir, saved_model_name), write_images=True, histogram_freq=1, update_freq=2000)
	earlystop = EarlyStopping(
		monitor='val_loss', patience=arg.early_stop_patience)

	steps_per_epoch = train_gen.steps_per_epoch / arg.log_freq
	model.fit_generator(
		train_gen(), validation_data=val_gen(),
		steps_per_epoch=steps_per_epoch, epochs=arg.epochs,
		validation_steps=val_gen.steps_per_epoch,
		use_multiprocessing=True,
		max_queue_size=100,
		workers=arg.workers,
		callbacks=[model_checkpoint, tensorboard, earlystop])
