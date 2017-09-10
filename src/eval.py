r"""Evaluation interface for detection models.
"""
import functools
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from src.configuration.configuration import Configuration
#from src.evaluation.evaluation import Evaluation
from src.configuration.configure import configure_files
from src.train_test.darknet import Darknet

def parse_args(args_list):
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--logstostderr', action='store_true', default=False, help='Whether or not to show the errors')
    parser.add_argument('--pipeline_config_path', type=str, action='store', default = 'pipeline.cfg', help='path to config')
    parser.add_argument('--model_path', action='store', help='Path to the model to evaluate')

    args = parser.parse_args(args_list)
    return args

def main(unused_argv):
    args = parse_args(unused_argv)
    conf = Configuration(args.pipeline_config_path) 
    # Overwrite the configuration parameters if necessary
    if args.model_path:
        conf.final_weights = args.model_path
    configure_files(conf)
    dk = Darknet()
    dk.launch_test(conf)

if __name__=='__main__':
    main(sys.argv[1:])
