import read_image as ri
import logging
import model
import modell
import os
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

IMG_DATASET_PATH = '/Users/melissa/Documents/446/proj/face_images_att'

# 


def main():
    '''
    Reads images from the dataset we are learning from.
    '''

    logger.info('Running main!')
    [imgs, face_ids] = ri.read_images(IMG_DATASET_PATH)

    # component_prec = []
    # numcom_list = []

    # for num_components in xrange(50, 375, 25):

    #     precision = []
    #     for i in xrange(30):
    #         mod = model.Model(IMG_DATASET_PATH, num_components)
    #         precision.append(mod.compute())

    #     component_prec.append(np.mean(precision))
    #     numcom_list.append(num_components)

    # print numcom_list
    # print component_prec

    mod = model.Model(IMG_DATASET_PATH, 320)
    mod.compute(k=1)
    #mod.plotEigenvectors()  # plots normalized Eigenvectors
    
    # plots the reconstruction of a specified image
    # mod.plotReconstruction('/Users/melissa/Documents/446/proj/face_images_att/s1/1.pgm')





    print('exit success')


if __name__ == '__main__':
    main()