import argparse
import tensorflow as tf
from preprocessing import preprocessing, l2_norm
from inference import frozen_function

#command line argument parser
def parse_args():

    #Create the parser 
    parser = argparse.ArgumentParser(description="Making inference on ArcFace", allow_abbrev=False)

    #Adding arguments 
    parser.add_argument('img_path',
                        type = str,
                        help = 'This is the path to the image ')
    
    parser.add_argument('pb_path',
                        type = str,
                        help = 'This is the path to the PB file ')

    args =parser.parse_args()

    return args

def embeddings(model):
    img_path = parse_args().img_path

    # This is needed for loading frozen pb file

    # frozen_func = frozen_function(pb_path)
    # new_img, fliped = preprocessing(img_path)

    # # Get predictions for test images
    # new_img_pred = frozen_func(x=tf.constant(new_img))[0]
    # fliped_pred = frozen_func(x=tf.constant(fliped))[0]

    new_img, fliped = preprocessing(img_path)

    # Get predictions for test images
    new_img_pred = model(new_img)
    fliped_pred = model(fliped)

    embeds = new_img_pred.numpy() + fliped_pred.numpy()
    embeds = l2_norm(embeds) 
    return embeds

if __name__ == '__main__':
    print("-" * 50)
    print("Making predictions on a Single Image ")


    pb_path = parse_args().pb_path
    # This is required for loading saveModelformat
    model = tf.keras.models.load_model(pb_path)


    embeds = embeddings(model)
    print("Done")
    print(embeds)
    