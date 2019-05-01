#!/usr/bin/python
#     file: test_with_unet.py
#   author: Ziyi Cui
#  created: April 28th, 2019
#  purpose: xxx

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import os
import sys
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
curr_path = os.getcwd()
sys.path.insert(0, curr_path + "/data_process")
from data_loader import KerasTestDataGenerator
from modeling import *

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
if __name__ == "__main__":
    model = unet()
    model.load_weights('modeling.hdf5')

    print('...start testing...')
    test_ds = KerasTestDataGenerator(
        test_folder=curr_path + '/data/test/tumor_001/')
    test_ds = test_ds()

    results = model.predict_generator(test_ds, steps=5, verbose=1)
    print('...end testing...')

    num_test_data = results.shape[0]
    prob = np.average(results.reshape(num_test_data, -1), axis=1)

    df = pd.read_csv(curr_path + '/data/test/tumor_001/test_index.csv')
    df['tumor_prob'] = pd.Series(prob, index=df.index)