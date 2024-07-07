#import necessary libraries
import os
import time
import random
import shutil
import argparse
import tensorflow as tf

#import necessary files

from src.losses import SoftmaxLoss
from src.models import ArcFaceModel
import src.data_generator as dataset
from src.utils import load_yaml, get_ckpt_inf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

#command line argument parser
def parse_args():

    #Create the parser 
    parser = argparse.ArgumentParser(
        description="Train Livliness model", 
        allow_abbrev=False)

    #Adding 3 arguments to access aws
    parser.add_argument('--train_tfrecord_path', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'),
                        help='Path to train tf record')

    parser.add_argument('--is_retraining', type=bool,
                        default=False,
                        help='Determine whether to retrain with different parameters')
    
    parser.add_argument('--model_dir', type=str,
                        help='Path to model folder',
                        default='/opt/ml/model')

    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to checkpoint folder',
                        default='/opt/ml/checkpoints')

    args =parser.parse_args()

    return args


if __name__ =='__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # To prevent tensorflow from allocating the 
    # totality of a GPU memory for training
    #set_memory_growth()
    
    #Look deep into there config(opt file) and also dont forget they ensemble pretrained
    #network trained on three training folds and with two initial seeds
    cfg = load_yaml('./config/arc_mbv2.yaml')
    
    #backbone architecture
    print("Loading model")
    model = ArcFaceModel(size=cfg['input_size'],
                        channels=3, 
                        num_classes=cfg['num_classes'], 
                        name='arcface_model', 
                        margin=0.5, 
                        logist_scale=64, 
                        embd_shape=cfg['embd_shape'], 
                        head_type=cfg['head_type'],
                        backbone_type=cfg['backbone_type'],
                        w_decay=cfg['w_decay'],
                        use_pretrain = True, 
                        training=True)


    #some model training parameters 
    dataset_len = cfg['num_samples']
    steps_per_epoch = dataset_len // cfg['batch_size']
    
    print("loading dataset")
    train_dataset = dataset.load_tfrecord_dataset(args.train_tfrecord_path, 
                                                  cfg['batch_size'],
                                                  binary_img= cfg['binary_img'],
                                                  is_ccrop= cfg['is_ccrop'])

    # Instantiate optimizer
    if args.is_retraining:
        print("Changing optimizer value")
        learning_rate = tf.constant(random.choice(cfg['base_lr']))
        print(f"Using a lerning rate of {learning_rate}")
    else:
        learning_rate = tf.constant(cfg['base_lr'][0])
        print(f"Using a learning rate of {learning_rate}")
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()

    # Load model
    checkpoint_path  = args.checkpoint_path + "/ckpt"
    if not os.path.exists(checkpoint_path):
        # Load model
        os.mkdir(f'{checkpoint_path}')
    
    ckpt_path = tf.train.latest_checkpoint('ckpt/peteryuX_ckpt')
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, loss = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, loss = 1, 1
    

    model.compile(optimizer=optimizer, loss=loss_fn) #Try true

    print('****** save ckpt file! ******')
    check_path = (f'{checkpoint_path}' + '/e_{epoch}_l_{loss}.ckpt')
    model_path = f'{args.checkpoint_path}' + "/SavedModel"
    log_dir=f'{args.checkpoint_path}' + "/logs/"
    print(check_path, model_path)


    print("****** convert to savedModel format ******")
    model.save(model_path)

    mc_callback = ModelCheckpoint(check_path,save_freq = cfg['save_steps'], verbose=1, save_weights_only=True)
    # tb_callback = TensorBoard(log_dir=log_dir, update_freq=cfg['batch_size'] * 5, profile_batch=0)
    # tb_callback._total_batches_seen = steps
    # tb_callback._samples_seen = steps * cfg['batch_size']
    callbacks = [mc_callback]
    # callbacks = [mc_callback, tb_callback]

    print("[*] training starting!")
    model.fit(train_dataset, epochs=cfg['epochs'], steps_per_epoch=steps_per_epoch, callbacks=callbacks, initial_epoch=epochs - 1)
    print("[*] training done!")

    print("****** convert to savedModel format ******")
    model.save(model_path)
    zip_path = f'{args.checkpoint_path}' + "/model_files/model.tar.gz"
    os.system(f'tar -czvf {zip_path} {model_path}')