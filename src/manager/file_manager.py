import os
import shutil

def save_weights(conf, hash):
    print(conf.final_weights)
    copied_file_name = os.path.join(conf.weights_folder,
                                    str(hash) + '.weights')
    shutil.copy(conf.final_weights, copied_file_name)


def load_weights(conf, hash):
    print('Should load weights')