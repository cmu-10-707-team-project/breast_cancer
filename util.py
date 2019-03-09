#!/usr/bin/python
#     file: util.py
#   author: Ziyi Cui
#  created: Mar.7th, 2019
#  purpose: Spring 2019, 10707, project

# # # # # # # # # # #
#   I M P O R T S   #
# # # # # # # # # # #
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
# # # # # # # # # # # # #
#   F U N C T I O N S   #

################################################################################
def valid_directory(parser, arg):
    if not os.path.exists(arg):
        parser.error('The directory \"' + str(arg) + '\" does not exist.')
    return arg

def extract_target_files(dir, ext):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    return [f for f in onlyfiles if f.endswith(ext)]

def np_to_pil(np_img):
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)
