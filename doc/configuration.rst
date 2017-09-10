Configuration
+++++++++++++

The configuration allows you to choose between the different models available in the pipeline. The main configuration file of the pipeline
defaults to 'pipeline.cfg'.

In this general pipeline you can specify a number of paths to essential folders to the pipeline :

[DEFAULT] section

* configFolder : the folder where the pipeline will look for configuration files
* dataFolder : the folder for all your datasets
* SourceData : the file in configFolder that describes the dataset. Within that file you can specify
    - classes : number of classes in the dataset
    - train : path to the training set (a list of images in a .txt file)
    - val : path to the validation set (similar)
    - names : path to the names of the classes
    - backup : a folder to save temporary checkpoint files
    - eval (optional): Evaluation method

One of the following must be defined

* SourceDarknet : the configuration file for Darknet
* SourceFrRCNN : the configuration file for Faster-RCNN

* weightsFolder : The folder where starting weights and final weights are saved
* Startweights : The initial weights for a model

[DATASETS] section

* List the names of the datasets to use in the pipeline. These must be folder paths from the data folder. The folder structure of one particular dataset must have a folder Annotations, JPEGImages and ImageSets

[AUGMENTATION] section

* AUGMENT : whether or not to use augmented datasets
* AUGMENTED_DATA : whether or not datasets have already been augmented