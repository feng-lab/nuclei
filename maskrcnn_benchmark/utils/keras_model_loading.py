# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import pickle
from collections import OrderedDict

import torch
import h5py


if __name__ == '__main__':
    f = h5py.File('/Volumes/fs3017/eeum/nuclei/models/deepretina_final.h5', mode='r', libver='latest')

    for layer_name in f.attrs['layer_names']:
        print(layer_name)
