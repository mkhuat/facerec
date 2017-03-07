# This file reads images in the current working directory.

import os
import sys
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


def read_single_image(filename, sz=None):
    """
    Reads a single image with the given file name.

    Args:
        filename: Name of file to read.

    Returns image: multidimensional image
    """

    try:

        # Open image
        img = Image.open(os.path.join(filename))
        
        # Convert image to black-and-white
        img = img.convert('L')

        # Resize to given size (if given)
        if (sz is not None):
            img = img.resize(self.sz, Image.ANTIALIAS)

        # Return image and metadata
        return img
    except IOError as e:
        logger.error('I/O error: {0}'.format(e))
    except:
        logger.error('Cannot open image')
    return None


def read_images(path):
    """
    Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
            Local path used: /Users/melissa/Documents/446/proj/face_images

    Returns:
        A list [images, id_nums]

            images: The images, which is a Python list of numpy arrays.
            id_nums: The corresponding labels (the unique id number of the person) in a Python list.
    """
    id_num = 0
    images, id_nums = [], []

    # Process individual files from subdirectories in given path
    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)    # path/subject_path/

            for imagename in os.listdir(subject_path):           # path/subject_path/imagename
                img = read_single_image(os.path.join(subject_path, imagename))
                
                if img is not None:
                    images.append(np.asarray(img, dtype='float64'))
                    id_nums.append(id_num)

            id_num += 1

    assert len(images) == len(id_nums)
    return [images, id_nums]
