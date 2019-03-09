#!/usr/bin/python
#     file: read_wsi.py
#   author: Ziyi Cui
#  created: Feb 25, 2019
#  purpose: Spring 2019, 10707, project

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import argparse

import datetime
import sys
import math
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import PIL
from PIL import Image
import multiprocessing

## my module
from util import extract_target_files,valid_directory
# # # # # # # # # # # # #
#   F U N C T I O N S   #
# # # # # # # # # # # # #
SCALE_FACTOR = 32
################################################################################
def main():
    args= get_args(sys.argv[1:])
    in_dir = args['in_directory']
    out_dir = args['out_directory']

    onlytif = extract_target_files(in_dir, 'tif')

    for tif in onlytif:
        sample_path = in_dir + tif
        output_path = out_dir + tif.split('.')[0] + '.jpg'
        wsi_to_jpg(sample_path,output_path)


################################################################################
def multiprocess_wsi_to_img():
    mum_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

def wsi_to_jpg(sample_path,output_path):
    slide  = OpenSlide(sample_path)
    w,h = slide .dimensions
    new_w = math.floor(w / SCALE_FACTOR)
    new_h = math.floor(h / SCALE_FACTOR)
    level = slide .get_best_level_for_downsample(SCALE_FACTOR)
    pil_img = slide .read_region((0, 0), level, slide .level_dimensions[level]).convert("RGB").resize((new_w,new_h),PIL.Image.BILINEAR)
    pil_img.save(output_path)

def get_args(args):
    parser = argparse.ArgumentParser(prog ='extract one resolution from WSI: x32')

    parser.add_argument('-din', '--in_directory', \
    type = lambda x: valid_directory(parser, x), \
    help = 'directory containing .tif files',\
    required = True)

    parser.add_argument('-dout', '--out_directory', \
    type = lambda x: valid_directory(parser, x), \
    help = 'directory to store the converted JPG files',\
    required = True)

    return vars(parser.parse_args(args))
################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # #
#   C A L L   T O   M A I N   F U N C T I O N   #
# # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == "__main__":
    main()