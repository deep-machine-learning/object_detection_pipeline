Model description
-----------------

The model can consist of one end-to-end detection model such as Faster-RCNN or Yolo. The model consists of several layers of convolutional layers followed by a detection layer. The implementation is in C for fast processing time. The model takes an image as input and outputs bounding boxes with probabilities.

Data related parameters : 

•	"classes" : the number of object classes to be detected

•	"train" : the path to a file with the path to the images to use for the training (created from the list of datasets at the configuration step)

•	"val" : "" for validation(created from the list of datasets at the configuration step)

•	"names" : the path to a file containing the names (created from a list in the configuration file)

•	"backup" : the path to a folder that will be used to save intermediate checkpoints for the training (created when using the makefile during installation)

Relevant model parameters : 
•	"batch_size": self-explanatory

•	"learning rate": self-explanatory

•	"subdivisions": Divides the batches by subdivision to reduce memory usage but only updates the weight after the whole batch has been processed
