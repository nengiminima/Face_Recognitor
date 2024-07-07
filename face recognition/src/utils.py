import yaml
import numpy as np
import tensorflow as tf
from absl import logging

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded

def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_l_')
    epochs = int(split_list[0])
    # batchs = int(split_list[-1].split('_l_')[0])
    loss =  split_list[1].split('.ckpt')[0]
    # steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, loss