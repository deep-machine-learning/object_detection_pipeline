Evaluation
++++++++++

The evaluation bit of the pipeline is essential to pick the model to be deployed, and to assess which modifications to
the configuration or the pipeline yield the best results. For our evaluation we use the "mean AP" measure that is commonly
used in object detection tasks and among others in the Pascal VOC challenges.

The evaluation is performed per class before it is averaged out on all the classes.

Intersection over Union
///////////////////////

Recall
//////

AP
//
