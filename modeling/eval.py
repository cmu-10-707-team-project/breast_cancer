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

	parser.add_argument('--model-name', type=str, required=True)
	parser.add_argument('--saved-model', type=str, required=True)
	parser.add_argument('--workers', type=int, default=os.cpu_count())

	parser.add_argument('--test-index-file-path', type=str, required=True)
	parser.add_argument('--test-folder', type=str, required=True)

	parser.add_argument('--lr', type=float, default=1e-3)

	parser.add_argument('--batch-size', type=int, default=32)

	arg = parser.parse_args()

	# use model to predict test data
	model = get_model(**arg.__dict__)
	model.load_weights(arg.saved_model)

	print ('...start testing...')
	test_ds = KerasDataGenerator(
		index_filepath=arg.test_index_file_path, input_folder=arg.test_folder,
		batch_size=arg.batch_size, labeled=True, mask=arg.model_name == 'unet')

	results = model.evaluate_generator(
		test_ds(), steps=test_ds.steps_per_epoch, verbose=1)

	print(results)
