import read_image as ri
import logging
import model
import os

logger = logging.getLogger(__name__)

logging.basicConfig(filename='log_' + os.path.basename(__file__),
                            filemode='w',
                            format='%(asctime)s %(name)s %(levelname)s:\t %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

IMG_DATASET_PATH = '/Users/melissa/Documents/446/proj/face_images_att'


def main():
    '''
    Reads images from the dataset we are learning from.
    '''

    logger.info('Running main!')
    [imgs, face_ids] = ri.read_images(IMG_DATASET_PATH)

    mod = model.Model('/Users/melissa/Documents/446/proj/face_images_att')
    mod.compute()
    mod.plotEigenvectors()  
    #mod.plotReconstruction()

    print('exit success')


if __name__ == '__main__':
    main()