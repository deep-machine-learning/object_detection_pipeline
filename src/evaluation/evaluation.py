from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import http.client
import json
from src.data_augmentation.image import Image
from src.data_augmentation.helpers import get_labelled_images, convert_yolo_opencv
from src.evaluation.helpers import calculate_intersection_over_union, calculate_average_precision
from collections import defaultdict
import datetime
import shutil
import os
import random

class Evaluation(object):
    def __init__(self, configuration=None):
        self.server_connection = http.client.HTTPConnection(configuration.api_server)
        self.test_set = configuration.test_set
        self.prediction_confidence_level = configuration.confidence_predictions
        self.list_classes = configuration.list_classes
        self.labels_predictions = {}
        self.mapped_labels_predictions = {}
        self.conf = configuration


    def get_labels_predictions(self, image):
        labels_predictions = {}
        for i, label in enumerate(image.labels.values()):
            xmin, ymin, xmax, ymax = convert_yolo_opencv(image.shape, float(label['x']), float(label['y']), float(label['w']), float(label['h']))
            object_class = int(label['class_id'])  # self.list_classes[int(label['class_id'])]
            if object_class in labels_predictions.keys():
                labels_predictions[object_class]['labels'].append([xmin, ymin, xmax, ymax])
            else:
                labels_predictions[object_class] = {}
                labels_predictions[object_class]['labels'] = []
                labels_predictions[object_class]['labels'].append([xmin, ymin, xmax, ymax])

        image = "{\n\"frames\":[{\"data\":\"" + image.convert_to_base64() + "\"\n}]\n}"
        self.server_connection.request("POST", "/resource", image)
        response = self.server_connection.getresponse()
        tags = json.loads(response.read().decode('utf-8'))['tags']
        for i, tag in enumerate(tags):
            if tag['confidence_level'] > self.prediction_confidence_level:
                object_class = self.list_classes.index(tag['name'])
                if object_class in labels_predictions.keys():
                    if 'predictions' in labels_predictions[object_class].keys():
                        labels_predictions[object_class]['predictions'].append([tag['top'], tag['left'], tag['bot'], tag['right'], tag['confidence_level']])
                    else:
                        labels_predictions[object_class]['predictions'] = []
                        labels_predictions[object_class]['predictions'].append([tag['top'], tag['left'], tag['bot'], tag['right'], tag['confidence_level']])
                else:
                    labels_predictions[object_class] = {}
                    labels_predictions[object_class]['predictions'] = []
                    labels_predictions[object_class]['predictions'].append([tag['top'], tag['left'], tag['bot'], tag['right'], tag['confidence_level']])
        self.labels_predictions = labels_predictions

    def map_labels_predictions(self, image_set):
        mapped_labels_predictions = defaultdict(dict)
        for i, item in enumerate(image_set.values()):
            image = Image(image_path=item[0], labels=item[1])

            self.get_labels_predictions(image)
            mapped_labels_predictions[i]['image_path'] = image.image_path
            mapped_labels_predictions[i]['objects'] = self.labels_predictions
        self.mapped_labels_predictions = mapped_labels_predictions

    def calculate_map(self):
        # for c, object_class in enumerate(self.list_classes):
        #     print(object_class)
        #     for i in self.mapped_labels_predictions:
        #         if c in self.mapped_labels_predictions[i]['objects'].keys():
        #             print(self.mapped_labels_predictions[i]['image_path'], self.mapped_labels_predictions[i]['objects'][c])
        #     print('='*40)
        for c, object_class in enumerate(self.list_classes):
            print(object_class)
            for image in self.mapped_labels_predictions:
                if c in self.mapped_labels_predictions[image]['objects'].keys():
                    calculate_average_precision(self.mapped_labels_predictions[image]['objects'][c])
            print('=' * 40)

    def evaluate(self):

        filename = 'comp4' + '_det_' + test + '_{:s}.txt'
        annopath = '{:s}'
        for obj_class in conf.list_classes:

            rec, prec, ap = voc_eval(detpath, annopath,
                 imagesetfile,
                 obj_class,
                 conf.results,
                 ovthresh=0.5)
            print('class : {0} \n rec {1}, prec {2}, ap {3}'.format, obj_class, rec, prec, ap)
        # labelled_images = get_labelled_images(self.test_set)
        # self.map_labels_predictions(labelled_images)
        # self.calculate_map()
        # self.calculate_map()

    def save(self):
        # Get weights file output from configurationdra
        final_weights_file = self.conf.final_weights #'{0}_final.weights'.format(self.conf.net_name)
        # Write contents to file determined by label name and conf
        copied_file_name = os.path.join(self.conf.weights, self.conf.output_label + str(datetime.datetime.now())+'.weights')
        shutil.copy(final_weights_file, copied_file_name)

