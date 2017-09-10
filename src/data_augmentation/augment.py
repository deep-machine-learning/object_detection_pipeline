from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.data_augmentation.image import Image
from src.data_augmentation.helpers import get_labelled_images, save_image
import shutil


def augment_datasets(configuration):
    '''
    Purpose: Main function to augment datasets from the list of datasets in configuration
    :param configuration:
    :return:
    '''
    for dataset_path in configuration.datasets_path:
        try:
            shutil.copytree(dataset_path, dataset_path + '_augmented')
            labelled_images = get_labelled_images(dataset_path)  # get dictionary of images with labels

            for item in labelled_images.values():
                img = Image(image_path=item[0], labels=item[1])
                for i in range(0, 10):
                    img.random_vertical_flip()
                    img.random_gaussian_blur()
                    img.random_gamma()
                    # img.display_image_with_labels()
                    save_image(dataset_path + '_augmented/', img, configuration.data_format)

        except Exception as e:
            print('Error: %s' % e)

    # update configuration with list of augmented datasets
    configuration.AUGMENTED_DATA = True
    configuration.datasets_list = [d + '_augmented' for d in configuration.datasets_list]
    configuration.datasets_path = [d + '_augmented' for d in configuration.datasets_path]
    print("-- Data Augmented --")
