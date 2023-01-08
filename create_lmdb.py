# In this file we create our lmdb (Lightning Memory-Mapped Database) and import all the images form the
# specified file and add them to our database using transaction in the form of binary stream.

import argparse # The argparse module makes it easy to write user-friendly command-line interfaces.
import pickle # Pickle in Python is primarily used in serializing and deserializing a Python object structure.
import cv2 # cv2 is the module import name for opencv-python.
import lmdb # This is a universal Python binding for the LMDB 'Lightning' Database
from path import Path # provides various classes representing file system paths with semantics appropriate for different operating systems

parser = argparse.ArgumentParser() # places the extracted data in a argparse
parser.add_argument('--data_dir', type=Path, required=True) # To add program arguments
args = parser.parse_args() # runs the parser and places the extracted data in a argparse

# 2GB is enough for IAM dataset
assert not (args.data_dir / 'lmdb').exists() # it creates the lmdb file
env = lmdb.open(str(args.data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 2) # open databse to import

# go over all png files
fn_imgs = list((args.data_dir / 'img').walkfiles('*.png'))

with env.begin(write=True) as txn: # txn: transaction
    for i, fn_img in enumerate(fn_imgs): # adds a counter to an iterable and returns it in a form of enumerating object
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE) # loads an image from the specified file
        basename = fn_img.basename() # used to get the base name in specified path
        txn.put(basename.encode("ascii"), pickle.dumps(img)) # create binary stream an put to database
env.close() # close the database
