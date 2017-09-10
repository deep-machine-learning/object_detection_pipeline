import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from src.configuration.configuration import Configuration
from src.configuration.search_configuration import SearchConfiguration
from src.evaluation.evaluation import Evaluation
from src.configuration.configure import configure_files
from src.data_augmentation.augment import augment_datasets
from src.train_test.darknet import Darknet
from src.optimizer.optimizer import Optimizer
import http.client


def main():
    conf = Configuration()

    if conf.AUGMENT:
        if not conf.AUGMENTED_DATA:
            augment_datasets(conf)  # augment datasets if it is required in configuration

    # if conf.search:
    #     search_conf = SearchConfiguration()
    #     opt = Optimizer(search_conf)
    #     opt.minimize()
    # else:
    # configure_files(conf)

    dk = Darknet()
    # dk.launch_train(conf)
    # dk.launch_test(conf)
        # dk.launch_test_with_recall(conf)

        # evaluation = Evaluation(conf)
        # evaluation.evaluate()
        # evaluation.save()


if __name__ == '__main__':
    main()
