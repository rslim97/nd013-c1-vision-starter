import argparse
import glob
import os
import random

import numpy as np
import shutil
from random import shuffle

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function

    file_paths=glob.glob(os.path.join(source,"*.tfrecord"))
    shuffle(file_paths)
    for _dir in ["train","val","test"]:
        dir_path=os.path.join(destination,_dir)
        os.makedirs(dir_path,exist_ok=True)

    # split out the training part, 0.8
    start=0
    end=start + int(0.8*len( file_paths))
    for file in file_paths[start:end]:
        dir_path=os.path.join(destination,"train")
        dest_path=os.path.join(dir_path,os.path.basename(file))
        shutil.move(file,dest_path)

    # split out the eval part, 0.1
    start=end
    end=start + int(0.1*len(file_paths))
    for file in file_paths[start:end]:
        dir_path=os.path.join(destination,"val")
        dest_path=os.path.join(dir_path,os.path.basename(file))
        shutil.move(file,dest_path)

    # split out the final part for testing
    for file in file_paths[end:]:
        dir_path=os.path.join(destination,"test")
        dest_path=os.path.join(dir_path,os.path.basename(file))
        shutil.move(file,dest_path)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)

# source: /app/project/data/processed
# destination: /app/project/data 