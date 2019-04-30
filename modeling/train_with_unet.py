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

from modeling.data_loader import KerasDataGenerator
from modeling import unet


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
	parser.add_argument('--model-name', type=str, default="modeling")
	parser.add_argument('--early-stop-patience', type=int, default=10)

	parser.add_argument('--workers', type=int, default=cpu_count())

	parser.add_argument('--train-index-file-path', type=str, required=True)
	parser.add_argument('--train-input-folder', type=str, required=True)
	parser.add_argument('--val-index-file-path', type=str, required=True)
	parser.add_argument('--val-input-folder', type=str, required=True)

	arg = parser.parse_args()
	
	train_gen = KerasDataGenerator(
		batch_size =arg.batch_size,
		index_filepath=arg.train_index_file_path,
		input_folder=arg.train_input_folder,
		is_train=True)

	val_gen = KerasDataGenerator(
		batch_size=arg.batch_size,
		index_filepath=arg.val_index_file_path,
		input_folder=arg.val_input_folder,
		is_train=False)

	model = unet()

	timestamp = datetime.now().strftime('%m-%d-%H%M%S')
	model_path = '{}_{}.hdf5'.format(arg.model_name, timestamp)
	model_checkpoint = ModelCheckpoint(
		'modeling.hdf5', monitor='loss',verbose=1, save_best_only=True)
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
