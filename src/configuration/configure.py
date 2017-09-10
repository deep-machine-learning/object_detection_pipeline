import random
import xml.etree.ElementTree as ET
import os
import imghdr


def configure_voc_data(configuration):
    '''Write the data configuration to the voc_data file (should be in temp)'''
    with open(configuration.voc_data, 'w') as f:
        f.write('classes = {0}{1}'.format(configuration.num_classes, '\n'))
        f.write('train = {0}{1}'.format(configuration.train_set, '\n'))
        f.write('valid = {0}{1}'.format(configuration.valid_set, '\n'))
        f.write('names = {0}{1}'.format(configuration.voc_names, '\n'))
        f.write('backup = {0}{1}'.format(configuration.backup, '\n'))
        f.close()


def configure_voc_names(configuration):
    '''Writes the names to the voc_names file (should be in data)'''
    with open(configuration.voc_names, 'w') as f:
        for c in configuration.list_classes:
            f.write(c + '\n')
        f.close()


def configure_yolo_cfg(configuration):
    '''Writes contents of the source_net_cfg_file into the net_cfg file while making sure the number of filters of yolo corresponds to the classes'''
    with open(configuration.source_net_cfg_file, 'r') as f:
        lines = f.readlines()
        # find the index of filters and classes in the file
        indicies = [i for i, s in enumerate(lines) if 'classes' in s or 'filters' in s]
        classesIndex = len(indicies) - 1
        lines[indicies[classesIndex - 1]] = 'filters={0}{1}'.format((configuration.num_classes + 4 + 1) * 5, '\n')  # change number of classes
        lines[indicies[classesIndex]] = 'classes={0}{1}'.format(configuration.num_classes, '\n')  # change the number of filters in the CONV layer above the region layer - (#classes + 4 + 1)*(5)
    with open(configuration.net_cfg, 'w') as f:
        f.writelines(lines)
        f.close()


def generate_val_train(configuration):
    '''Performs train/ validation split within a dataset and writes the files for each set to ImageSets/Main/{train or val}.txt'''
    for folder in configuration.datasets_path:
        imgfiles = []
        for file in os.listdir(folder + "/JPEGImages"):
            if configuration.data_format == ".":
                configuration.data_format += imghdr.what(os.path.join(folder, 'JPEGImages', file))
            imgfiles.append(file)
        random.shuffle(imgfiles)
        train = int(len(imgfiles) * 0.8)
        traindata = imgfiles[:train]
        valdata = imgfiles[-(len(imgfiles) - train):]

        with open("{0}/ImageSets/Main/train.txt".format(folder), mode="w", encoding="utf-8") as myfile:
            for line in traindata:
                myfile.write(os.path.splitext(line)[0] + "\n")
            myfile.close()

        with open("{0}/ImageSets/Main/val.txt".format(folder), mode="w", encoding="utf-8") as myfile:
            for line in valdata:
                myfile.write(os.path.splitext(line)[0] + "\n")
            myfile.close()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(configuration, dataset, image_id):
    '''Convert annotations from Pascal VOC to darknet format. If there are objects in the image, return True'''
    in_file = open('{0}/Annotations/{1}.xml'.format(dataset, image_id))
    out_file = open('{0}/labels/{1}.txt'.format(dataset, image_id), 'w')
    # print('{0}/Annotations/{1}.xml'.format(dataset, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    number_of_objects = 0
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in configuration.list_classes or int(difficult) == 1:
            continue
        cls_id = configuration.list_classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        number_of_objects += 1

    return number_of_objects > 0




def prepare_datasets(configuration):
    '''Loops through the training and validation sets of the datasets, convert them to the proper format and adds them to the main training and validation sets'''
    for dataset in configuration.datasets_path:
        if not os.path.exists('{0}/labels/'.format(dataset)):
            os.makedirs('{0}/labels/'.format(dataset))

    for subset in ['train', 'val']:
        l = []
        for dataset in configuration.datasets_path:
            for image_id in open('{0}/ImageSets/Main/{1}.txt'.format(dataset, subset), 'r').read().strip().split():
                if convert_annotation(configuration, dataset, image_id):
                    l.append(dataset + '/JPEGImages/' + image_id + configuration.data_format + '\n')

        with open('{0}.txt'.format(os.path.join(configuration.data_folder, subset)), 'w') as f:
            f.writelines(l)


def configure_files(configuration):
    '''Configure data file, class names files and network file (s)
     Then generate train/val sets for all datasets
     Finally, put train/val sets together and converts annotations to darknet format'''
    configure_voc_data(configuration)
    configure_voc_names(configuration)
    configure_yolo_cfg(configuration)
    print('-- Config files ready --')
    generate_val_train(configuration)
    print('-- Datasets ready --')
    prepare_datasets(configuration)
    print('-- Configuration Done --')
