from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from model.test import test_net
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.my_factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

import copy

class FasterRCNN(object):
    '''Class used as interface to Faster-RCNN'''

    def __init__(self):
        ''''''
        pass

    def launch_train(self, conf):
        '''
        
        '''
        args = {}
        args['cfg_file'] = conf.frcnn_cfg
        args['weight'] = conf.starting_weights
        args['imdb_name'] = conf.train_set
        args['imdbval_name'] = conf.valid_set
        args['max_iters'] = conf.iters
        args['tag'] = conf.frcnn_tag
        args['net'] = conf.frcnn_net
        args['set_cfgs'] = None

        print('Called with args:')
        print(args)

        if args['cfg_file'] is not None:
            cfg_from_file(args['cfg_file'])
        if args['set_cfgs'] is not None:
            cfg_from_list(args['set_cfgs'])

        print('Using config:')
        pprint.pprint(cfg)

        np.random.seed(cfg.RNG_SEED)

        # train set
        imdb, roidb = combined_roidb(args['imdb_name'], conf)
        print('{:d} roidb entries'.format(len(roidb)))

        # output directory where the models are saved
        output_dir = conf.backup_folder #get_output_dir(imdb, args.tag)
        print('Output will be saved to `{:s}`'.format(output_dir))

        # tensorboard directory where the summaries are saved during training
        tb_dir = conf.backup_folder # get_output_tb_dir(imdb, args.tag)
        print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

        # also add the validation set, but with no flipping images
        orgflip = cfg.TRAIN.USE_FLIPPED
        cfg.TRAIN.USE_FLIPPED = False
        _, valroidb = combined_roidb(args['imdbval_name'], conf)
        print('{:d} validation roidb entries'.format(len(valroidb)))
        cfg.TRAIN.USE_FLIPPED = orgflip
        if args['net'] == 'vgg16':
            net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
        elif args['net'] == 'res50':
            net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=50)
        elif args['net'] == 'res101':
            net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=101)

        # load network
        elif args['net'] == 'res152':
            net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=152)
        elif args['net'] == 'mobile':
            net = mobilenetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH)
        else:
            raise NotImplementedError

        train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
                  pretrained_model=args['weight'],
                  max_iters=args['max_iters'])




    def launch_test(self, conf, hash_model):
        '''
        '''
        args = {}
        args['cfg_file'] = conf.frcnn_cfg
        args['weight'] = conf.starting_weights
        args['model'] = hash_model
        args['imdb_name'] = conf.valid_set
        args['comp_mode'] = False
        args['tag'] = conf.frcnn_tag
        args['net'] = conf.frcnn_net
        args['set_cfgs'] = None
        args['max_per_image'] = 5

        print('Called with args:')
        print(args)

        if args['cfg_file'] is not None:
            cfg_from_file(argsargs['cfg_file'])
        if args['set_cfgs'] is not None:
            cfg_from_list(args['set_cfgs'])

        print('Using config:')
        pprint.pprint(cfg)

        # if has model, get the name from it
        # if does not, then just use the inialization weights
        if args['model']:
            filename = os.path.splitext(os.path.basename(args['model']))[0]
        else:
            filename = os.path.splitext(os.path.basename(args['weight']))[0]

        tag = args['tag']
        tag = tag if tag else 'default'
        filename = tag + '/' + filename

        # TODO This is really bad but it works, I'm sincerely sorry
        conf_copy = copy.deepcopy(conf)
        conf_copy.train_set = conf_copy.valid_set
        imdb = get_imdb(args['imdb_name'],  conf_copy)
        print(args['imdb_name'])
        imdb.competition_mode(args['comp_mode'])

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        # init session
        sess = tf.Session(config=tfconfig)
        # load network
        if args['net'] == 'vgg16':
            net = vgg16(batch_size=1)
        elif args['net'] == 'res50':
            net = resnetv1(batch_size=1, num_layers=50)
        elif args['net'] == 'res101':
            net = resnetv1(batch_size=1, num_layers=101)
        elif args['net'] == 'res152':
            net = resnetv1(batch_size=1, num_layers=152)
        elif args['net'] == 'mobile':
            net = mobilenetv1(batch_size=1)
        else:
            raise NotImplementedError

        # load model
        net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                                anchor_scales=cfg.ANCHOR_SCALES,
                                anchor_ratios=cfg.ANCHOR_RATIOS)

        if args['model'] :
            print(('Loading model check point from {:s}').format(args['model']))
            saver = tf.train.Saver()
            saver.restore(sess, args['model'])
            print('Loaded.')
        else:
            print(('Loading initial weights from {:s}').format(args['weight']))
            sess.run(tf.global_variables_initializer())
            print('Loaded.')

        test_net(sess, net, imdb, filename, max_per_image=args['max_per_image'])

        sess.close()

    def launch_test_with_recall(self, conf):
        '''Launches testing using darknet configuration files specified in the configuration
        The recall and average loss at every batch is printed
        '''
        pass


def combined_roidb(imdb_names_txt, conf):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name, conf)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return imdb, roidb

    # roidbs = [get_roidb(s) for s in imdb_names]
    # roidb = roidbs[0]
    # if len(roidbs) > 1:
    #     for r in roidbs[1:]:
    #         roidb.extend(r)
    #     tmp = get_imdb(imdb_names.split('+')[1])
    #     imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    # else:
    #     imdb = get_imdb(imdb_names)
    # return imdb, roidb

    imdb, roidb = get_roidb(imdb_names_txt)
    return imdb, roidb

