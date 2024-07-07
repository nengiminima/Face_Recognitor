# !pip install gdown
# !pip install mxnet
import os
# import io
# import cv2
import tqdm
import glob
import random
import argparse
# import numpy as np
# import mxnet as mx
# from PIL import Image
# from scipy import misc
import tensorflow as tf
from absl import logging

#command line argument parser
def parse_args():

    #Create the parser 
    parser = argparse.ArgumentParser(
        description="Train Livliness model", 
        allow_abbrev=False)

    #Adding 3 arguments to access aws
    parser.add_argument('--dataset_path', type=str,
                        default=os.environ.get('SM_ZIP_DEST_DIR'),
                        help='Path to images')
    
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to checkpoint folder',
                        default='/opt/ml/checkpoints')

    parser.add_argument('--tfrecord_name', type=str,
                        default=os.environ.get('SM_TFRECORD_NAME'),
                        help='Tfrecord name')

                        


    args =parser.parse_args()

    return args



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(args):
    dataset_path = f'{args.dataset_path}'
    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    samples = []
    logging.info('Reading data list...')
    for id_name in tqdm.tqdm(os.listdir(dataset_path)):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, id_name, filename))
    random.shuffle(samples)

    logging.info('Writing tfrecord file...')
    outputpath = f'{args.checkpoint_path}' + '/' + f'{args.tfrecord_name}' 
    with tf.io.TFRecordWriter(outputpath) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id_name),
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())


if __name__ =='__main__':
    args = parse_args()
    main(args)
