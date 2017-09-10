from hyperopt import fmin, tpe, Trials, hp, STATUS_OK

from src.configuration.configuration import Configuration
from src.evaluation.evaluation import Evaluation
from src.configuration.configure import configure_files
from src.train_test.darknet import Darknet


class Optimizer(object):

    def __init__(self, search_conf):
        '''Object acting as an interface to hyperopt'''
        self.search_configuration = search_conf
        self.hyper_params_space = {}

    def minimize(self):
        self.define_search_space()
        trials = Trials()
        best = fmin(objective_function, self.hyper_params_space, algo=tpe.suggest, trials=trials, max_evals=2)
        print(trials.trials[0])

    def define_search_space(self):

        self.hyper_params_space = {}
        self.hyper_params_space['AUGMENT'] = hp.choice('AUGMENT', [0, 1])


def objective_function(hyper_parameters):
    conf = Configuration()
    print(hyper_parameters)

    if hyper_parameters['AUGMENT'] == 0:
        conf.AUGMENT = False
    else:
        conf.AUGMENT = True

    configure_files(conf)
    # dk = Darknet()
    # dk.launch_train(conf)

    evaluation = Evaluation(conf)

    return {'loss': evaluation.evaluate(), 'status': STATUS_OK}
