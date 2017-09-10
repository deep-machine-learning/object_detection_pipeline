Installation
============

As of now, this pipeline works using the Darknet implementation of Yolo. Darknet was expanded by deep_machine_learnin to include a REST API. We in turn changed the compilation process and made a shared archive that allows us to use Darknet in Python.

Compiling Darknet-cpp
---------------------
The user can clone the below git repository
<https://github.com/deep-machine-learning/restful-yolo>

git clone https://github.com/deep-machine-learning/restful-yolo.git

Follow the instructions in the README.md file to install dependencies and compile objects. Once this is done, use the command::

make libdarknet.so

to create the libdarknet.so file which we will use later.

Getting the object_detection_pipeline
-------------------------

Clone the repository for this project, it can be found on 
<https://github.com/SofianHamiti/object_detection_pipeline>

git clone https://github.com/SofianHamiti/object_detection_pipeline.git

Copy the file libdarknet.so from Darknet-cpp

You're set! In order to run the pipeline, cd into object_detection_pipeline and type

python src/main.py

This will launch the pipeline with the default configuration.