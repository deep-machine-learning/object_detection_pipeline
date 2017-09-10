Data augmentation
/////////////////

The process of data augmentation aims at making the model more robust by modifying existing images in the dataset, for example by flipping them vertically, bluring them, adjusting the brightness, pixel intensity or gamma component of the image. Another dataset is then created in the list of datasets and is available for training.

Darknet automatically normalizes images. See `reference <https://groups.google.com/forum/#!searchin/darknet/normalize|sort:relevance/darknet/cjUmqL2Eb-s/RpHRywh_AwAJ>`_ (see as well image.c, function "ipl_to_image" called from load_image_cv)