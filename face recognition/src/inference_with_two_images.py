import argparse
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
# from inference import frozen_function
from preprocessing import preprocessing, l2_norm, load_tflite


#command line argument parser
def parse_args():

    #Create the parser 
    parser = argparse.ArgumentParser(description="Making inference on ArcFace", allow_abbrev=False)

    #Adding arguments 
    parser.add_argument('img_path1',
                        type = str,
                        help = 'This is the path to the image ')

    parser.add_argument('img_path2',
                        type = str,
                        help = 'This is the path to the image ')
    
    parser.add_argument('pb_path',
                        type = str,
                        help = 'This is the path to the PB file ')
    
    parser.add_argument('threshold',
                        type = float,
                        help = 'This is the number use to determine similar face ')
    
    parser.add_argument('is_tflite',
                        type=eval, 
                        choices=[True, False], 
                        default=False,
                        help = 'This specifies the model format ')

    args =parser.parse_args()

    return args

def embeddings(img_path, model, detector, output_details = None, tflite = False):
    # This is needed for loading frozen pb file

    new_img, fliped = preprocessing(img_path,detector)

    # Get predictions for test images
    new_img_pred = model(new_img)
    fliped_pred = model(fliped)

    embeds = new_img_pred.numpy() + fliped_pred.numpy()
    embeds = l2_norm(embeds) 
    return embeds


def dist_metric(embeds1, embeds2):
    thres = parse_args().threshold
    #l2 distance (Euclidean distance)
    diff = np.subtract(embeds1, embeds2)
    dist = np.sum(np.square(diff), 1)
    if dist > thres:
        return False, dist
    else:
        return True, dist

if __name__ == '__main__':

    img_path1 = parse_args().img_path1
    img_path2 = parse_args().img_path2

    print("-" * 50)
    print("Compare 2 images")

    pb_path = parse_args().pb_path
    detector = MTCNN()

    is_tflite = parse_args().is_tflite
    print(is_tflite)
    if is_tflite:
        model, output_details = load_tflite(pb_path)
        embeds1 = embeddings(img_path1, model, detector, output_details, is_tflite)
        embeds2 = embeddings(img_path2, model, detector, output_details, is_tflite)
    else:
        model = tf.saved_model.load(pb_path)
        embeds1 = embeddings(img_path1, model, detector)
        embeds2 = embeddings(img_path2, model, detector)

    boolval, dist = dist_metric(embeds1, embeds2)

    print(boolval, dist)