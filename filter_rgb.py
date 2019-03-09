#!/usr/bin/python
#     file: filter_rgb.py
#   author: Ziyi Cui
#  created: Mar.7th, 2019
#  purpose: Spring 2019, 10707, project

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse
import numpy as np
import PIL
from PIL import Image
import sys
## my module
from util import extract_target_files,valid_directory,np_to_pil

sys.path.insert(0, '/wsi')
from filter import apply_image_filters

# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
################################################################################
def main():
    args= get_args(sys.argv[1:])
    in_dir = args['in_directory']
    out_dir = args['out_directory']

    onlyjpg = extract_target_files(in_dir, 'jpg')
    

    for jpg in onlyjpg:
        sample_path = in_dir + jpg
        output_path = out_dir + 'filtered_'+ jpg
        pil_img = Image.open(sample_path)
        np_img = np.asarray(pil_img)
        filtered_np_img = apply_image_filters(np_img, slide_num=None, info=None, save=False, display=False)
        filtered_pil_img = np_to_pil(filtered_np_img)
        filtered_pil_img.save(output_path)
        
        # tile the image


    #apply_filters_to_image_list(image_num_list, save, display)
    #apply_filters_to_image_range(start_ind, end_ind, save, display)
    #singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)
    #multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)
################################################################################
def get_args(args):
    parser = argparse.ArgumentParser(prog ='filter jpg images at different channels')

    parser.add_argument('-din', '--in_directory', \
    type = lambda x: valid_directory(parser, x), \
    help = 'directory containing jpg files',\
    required = True)

    parser.add_argument('-dout', '--out_directory', \
    type = lambda x: valid_directory(parser, x), \
    help = 'directory to store filtered jpg files',\
    required = True)

    #parser.add_argument('-tile_out', '--tile_directory', \
    #type = lambda x: valid_directory(parser, x), \
    #help = 'directory to store tiles information',\
    #default = 0)

    return vars(parser.parse_args(args))
# # # # # # # # # # # # # # # # # # # # # # # # #
#   C A L L   T O   M A I N   F U N C T I O N   #
# # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":
    main()