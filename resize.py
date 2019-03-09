# convert sizes of raw input data into a same size
# https://gist.github.com/ihercowitz/642650/f01986c0b1ebd04be588b196eb3ffefe9853e113
from PIL import Image
from resizeimage import resizeimage
import os

def resizeImage(dir, infile, output_dir="", size=(768,1024)):
     outfile = os.path.splitext(infile)[0]
     extension = os.path.splitext(infile)[1]

     if (cmp(extension, ".jpg")):
        return

     if infile != outfile:
        #try :
        im = Image.open(dir+"/"+data_dir+infile)
        # scale
        im.thumbnail(size, Image.ANTIALIAS)
        # crop
        convert = resizeimage.resize_cover(im, (450, 1024))
        convert.save(output_dir+outfile+extension,"JPEG")
        #except IOError:
        #    print("cannot reduce image for {}".format(infile))

def cmp(a, b):
    return (a > b) - (a < b)

if __name__=="__main__":
    data_dir = "original_train/"
    dir = os.getcwd()
    output_dir = "resized_train/"

    if not os.path.exists(os.path.join(dir,output_dir)):
        os.mkdir(output_dir)

    for file in os.listdir(data_dir):
        resizeImage(dir, file, output_dir)