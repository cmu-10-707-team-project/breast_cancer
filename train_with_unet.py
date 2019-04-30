#!/usr/bin/python
#     file: train_with_unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import os 
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.getcwd()
sys.path.insert(0, curr_path + "/data_process")
from data_loader import KerasDataGenerator
from unet import *

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
if __name__=="__main__":
	ds = KerasDataGenerator(batch_size =32,
			index_filepath= curr_path+'/data/input/tumor_001/index.csv',\
	        input_folder = curr_path+'/data/input/',\
	        is_train=True)
	ds = ds()

	model = unet()
	model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
	model.fit_generator(ds,steps_per_epoch=8,epochs=1,callbacks=[model_checkpoint])

	#test_ds = KerasDataGenerator()
	#results = model.predict_generator(test_ds,30,verbose=1)
	#saveResult("data/test",results)