import argparse
import tensorflow as tf
import onnxruntime as ort
from mtcnn.mtcnn import MTCNN
from preprocessing import preprocessing, l2_norm, LoadOnnxModel

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

    parser.add_argument('landmark_path',
                        type = str,
                        help = 'This is the path to the lannkmark ONNX file ')

    args =parser.parse_args()

    return args

def embeddings(model, face_model_path, onnx_face_runtime):
    img_path = parse_args().img_path

    detector = MTCNN()
    new_img, fliped = preprocessing(img_path, detector, face_model_path, onnx_face_runtime, validate_face=True)

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

    face_model_path = parse_args().landmark_path
    face_shapes_model = LoadOnnxModel(face_model_path)
    onnx_face_runtime = ort.InferenceSession(face_model_path)


    embeds = embeddings(model, face_model_path, onnx_face_runtime)
    print("Done")
    print(embeds)
    