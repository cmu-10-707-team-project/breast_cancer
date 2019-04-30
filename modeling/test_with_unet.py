#!/usr/bin/python
#     file: test_with_unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse
from datetime import datetime
import os 
import sys
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from modeling.data_loader import KerasTestDataGenerator
from modeling import *

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
if __name__=="__main__":
	# parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--tensorboard-dir', type=str, default='data/log')
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--workers', type=int, default=cpu_count())

	parser.add_argument('--test-index-file-path', type=str, required=True)
	parser.add_argument('--test-folder', type=str, required=True)

	arg = parser.parse_args()

	# use model to predict test data
	model = unet()
	timestamp = datetime.now().strftime('%m-%d-%H%M%S')
	model.load_weights(arg.model)
	tensorboard = TensorBoard(
		log_dir=arg.tensorboard_dir, write_grads=False, write_images=False)

	print ('...start testing...')
	test_ds = KerasTestDataGenerator(test_folder = arg.test_folder)
	df = pd.read_csv(arg.test_index_file_pth)
	num_test_data = df.shape[0]
	results = model.predict_generator(test_ds(),steps=num_test_data,verbose=1)
	print ('...end testing...')

	# update the index file
	#num_test_data= results.shape[0]
	prob = np.average(results.reshape(num_test_data,-1),axis=1)
	df['tumor_prob'] = pd.Series(prob,index=df.index)
	df.to_csv(arg.test_index_file_pth, sep=',',index = False)