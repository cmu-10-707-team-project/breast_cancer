#!/usr/bin/python
#     file: eval.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse
import os

from modeling.model_factory import get_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from modeling.data_loader import KerasDataGenerator

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
if __name__=="__main__":
	# parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--tensorboard-dir', type=str, default='data/log')
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--saved-model', type=str, required=True)
	parser.add_argument('--workers', type=int, default=cpu_count())

	parser.add_argument('--test-index-file-path', type=str, required=True)
	parser.add_argument('--test-folder', type=str, required=True)

	arg = parser.parse_args()

	# use model to predict test data
	model = get_model(model_name=arg.model_name)
	model.load_weights(arg.model)

	print ('...start testing...')
	test_ds = KerasDataGenerator(
		index_filepath=arg.test_index_file_path, input_folder=arg.test_folder,
		labeled=True, mask=arg.model == 'unet')

	results = model.evaluate_generator(
		test_ds(), steps=test_ds.steps_per_epoch, verbose=1)

	print(results)
