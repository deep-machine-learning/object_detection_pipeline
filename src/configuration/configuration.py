from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from configparser import ConfigParser
from collections import OrderedDict
from itertools import chain
import hashlib

root = os.path.abspath("{0}/../../../".format(__file__))
# cfg_folder = "{0}/cfg/".format(root)
# data_folder = "{0}/data/".format(root)


class Configuration(object):
    '''
        A class containing all the information necessary to generate / modify

        * The names file : in self.voc_names (default : root/data/voc.names)

        * The network definition file : in self.net_cfg (default : root/cfg/yolo-voc.cfg)

        * The data definition file : in self.voc_data (default : root/cfg/voc.data)

        The class also contains information required to run the pipeline, including but not limited to :

        * starting_weights : the weights to load at the beginning of the training
        
        * final_weights : where the final weights are saved
        
        Modify the source code to use your own datasets and classes
    '''
    def __init__(self, pipeline_file='pipeline.cfg'):
        '''Creates a configuration object and parses the file pipeline.cfg. Also parses different files for model
        configuration and so on'''

        # folders with images and labels
        self.global_conf = ConfigParser(allow_no_value=True)
        self.global_conf.optionxform = str
        self.global_conf.readfp(open(pipeline_file))

        self.config_folder = self.global_conf['DEFAULT']['configFolder']
        self.data_folder = self.global_conf['DEFAULT']['dataFolder']
        self.weights_folder = self.global_conf['DEFAULT']['weightsFolder']
        self.source_data_file = os.path.join(self.config_folder, self.global_conf['DEFAULT']['SourceData'])
        self.source_net_cfg_file = os.path.join(self.config_folder, self.global_conf['DEFAULT']['SourceDarkNet'])
        self.starting_weights = os.path.join(self.weights_folder, self.global_conf['DEFAULT']['StartWeights'])

        self.datasets_list = []
        for ds in self.global_conf['DATASETS']:
            if not ds in self.global_conf['DEFAULT']:
                self.datasets_list.append(ds)

        self.datasets_path = [os.path.join(self.data_folder, p) for p in self.datasets_list]

        # dataset format
        self.data_format = ".jpg"

        # Data Augmentation
        self.AUGMENT = self.global_conf['AUGMENTATION'].getboolean('AUGMENT')
        self.AUGMENTED_DATA = self.global_conf['AUGMENTATION'].getboolean('AUGMENTED_DATA')

        self.list_classes = []
        self.num_classes = 0
        self.train_set = os.path.join(self.data_folder, 'train.txt')
        self.valid_set = os.path.join(self.data_folder, 'val.txt')
        self.backup_folder = None

        self.voc_names = None
        self.parse_data(self.source_data_file)
        self.backup = self.backup_folder
        # Output configuration files used directly by darknet
        self.temp_folder = 'temp'
        self.net_name = 'net'
        self.net_cfg = os.path.join(self.temp_folder, self.net_name+'.cfg')

        self.voc_data = '{0}/temp/data.data'.format(root)
        # voc_label parameters
        self.voc_label = '{0}voc_label.py'.format(self.data_folder)

        # where darknet will save the final weights
        self.final_weights = os.path.join(self.backup_folder, self.net_name+'_final.weights')

        #   evaluation
        self.test_set = '{0}TEST'.format(self.data_folder)
        self.api_server = "10.106.144.236:1984"
        self.confidence_predictions = 30

        # Define the filenames to save the weights (weights/self.output_label.weights) and some results
        self.output_label = 'output'

        # Search
        self.search = False

        self.net_config = None
        self.parse_net(self.source_net_cfg_file)

        # TODO : dehardcode this
        self.frcnn_cfg = None
        self.iters = 1000
        self.frcnn_tag = 'test'
        self.frcnn_net = 'vgg16'
        self.devkit_path = 'VOCdevkit'
        self.results = 'results'

    def parse_data(self, filename):

        data_conf = ConfigParser()
        with open(filename) as fp:
            # Tricking configparser into thinking there is a header
            fp = chain(('[top]',), fp)
            data_conf.readfp(fp)

        self.backup_folder = data_conf['top']['backup']
        self.voc_names = os.path.join(self.data_folder, data_conf['top']['names'])

        names_conf = ConfigParser(allow_no_value=True)
        with open(self.voc_names) as names_fp:
            names_fp = chain(('[top]',), names_fp)
            names_conf.readfp(names_fp)

        self.list_classes = []
        for ds in names_conf['top']:
            self.list_classes.append(ds)

        # voc.data parameters
        self.num_classes = len(self.list_classes)


    def __deepcopy__(self, memodict={}):
        return Configuration()

    def __repr__(self):

        ordered_d = OrderedDict(sorted(self.__dict__.items(), key = lambda t: t[0]))
        # print(repr(ordered_d))
        return repr(ordered_d)

        # voc.names parameters

    def get_hash(self):
        # TODO : the configuration doesn't return the same thing if it's called twice
        print()
        return hashlib.md5(self.__repr__().encode('utf-8')).hexdigest()

    def parse_net(self, filename):

        net_config = []
        with open(filename) as fp:
            section = ''
            for line in fp.readlines():
                if (line == '') | (line == '\n'):
                    if len(section_dict) > 0:
                        net_config.append({section: section_dict})
                    section = ''
                    section_dict = {}

                else:
                    if (line[0] == '[') & (line[-2] == ']'):
                        section = line.split('[')[1].split(']')[0]
                        section_dict = {}
                    else:
                        if len(line.split('=')) == 2:
                            sec_property, sec_attr = line.split('=')
                            section_dict[sec_property] = sec_attr

        self.net_config = net_config

    def step(self, hash):
        '''This function changes the starting weights of the model to hash+.weights'''
        self.starting_weights = os.path.join(self.weights_folder, hash+'.weights')
