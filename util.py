import os.path
import json
import numpy as np

from tensorflow_federated.python.simulation.hdf5_client_data import HDF5ClientData


def load_data():
    fileprefix = 'fed_emnist_digitsonly'
    dir_path = '/Users/mac/Documents/vis/TFF/fed_emnist_digitsonly'
    train_client_data = HDF5ClientData(
        os.path.join(dir_path, fileprefix + '_train.h5'))
    test_client_data = HDF5ClientData(
        os.path.join(dir_path, fileprefix + '_test.h5'))

    return train_client_data, test_client_data


# json encoder
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_as_json(data, file_name):
    result = json.dumps(data, cls=MyEncoder)
    with open(file_name, 'w') as f:
        json.dump(result, f)