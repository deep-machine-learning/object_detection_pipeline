from ctypes import *


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Darknet(object):
    '''Class used as interface to Darknet'''

    def __init__(self):
        '''Loads the shared archive compiled from darknet sources'''
        self.lib = CDLL("libdarknet.so", RTLD_GLOBAL)

    def launch_train(self, conf):
        '''Launches training using darknet configuration files specified in the configuration
        

        Positional arguments:
        conf -- a Configuration object that contains all the necessary file names to run Darknet. Note that 
        configure(conf) must be called before this function to make sure the training is performed with the 
        parameters specified in the configuration. Weight files are saved to conf.weights
        '''

        train = self.lib.train_detector_api
        train.argtypes = [c_char_p,c_char_p,c_char_p,c_int]
        train.restype = c_void_p
        datacfg = conf.voc_data
        net_cfg = conf.net_cfg
        weights = conf.starting_weights
        print(datacfg)
        print(net_cfg)
        print(weights)
        train(datacfg.encode('ascii'), net_cfg.encode('ascii'), weights.encode('ascii'),0)

    def launch_test(self, conf):
        '''Launches testing using darknet configuration files specified in the configuration
            
        Positional arguments:
        conf -- a Configuration object that contains all the necessary file names to run Darknet. Note that 
        configure(conf) must be called before this function to make sure the testing is performed with the 
        parameters specified in the configuration. Result files are saved to results/
        '''
        test = self.lib.validate_detector_api
        test.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p]
        test.restype = c_void_p
        datacfg = conf.voc_data
        net_cfg = conf.net_cfg
        weights = conf.final_weights
        out_file = conf.validation_output_path
        test(datacfg.encode('ascii'), net_cfg.encode('ascii'), weights.encode('ascii'), None)

    def launch_test_with_recall(self, conf):
        '''Launches testing using darknet configuration files specified in the configuration
        The recall and average loss at every batch is printed
        '''
        test = self.lib.validate_detector_recall_api
        test.argtypes = [c_char_p, c_char_p, c_char_p]
        test.restype = c_void_p
        net_cfg = conf.net_cfg
        weights = conf.final_weights
        validation_file = conf.valid_set
        test(net_cfg.encode('ascii'), weights.encode('ascii'), validation_file.encode('ascii'), None)

    def save_weights(self, conf):
        pass

