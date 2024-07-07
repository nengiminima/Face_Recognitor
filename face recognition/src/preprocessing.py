import cv2
import onnx
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

def showImageWithLandmark(image, landmark):
        """
        A description of the positions of the landmarks;
        Index ranges:
        Right face side - [0 - 6]
        Chin    - [7 - 9]
        Left face side - [10 - 16]
        Left eye brow - [17 - 21]
        Right eye brow - [22 - 26]
        Nose middle line - [27- 30]
        Nose under line - [31 - 35]
        Right eye points - [36 - 41]
        Left eye points - [42 - 47]
        Upper lips - [48 - 54]
        Lower lips - [55 - 59]
        Edge of mouth right - [60]
        Mouth inline - [60 - 67]
        Edge of mouth left - [64]
        """
        img_ = image.copy()
        
        # Check where the eye landmakarks are landmark = landmark[63:65, :]
        for x, y in landmark:
            cv2.circle(img_, (int(x), int(y)), 3, (0, 255, 0), -1)
        return img_

def LoadOnnxModel(model_filename):
    if not model_filename.endswith(".onnx"):
        raise ValueError("Onnx model path incorrect")
    return onnx.load(model_filename)

def projectLandmarks(landmarks, result):
    """Project the predicted landmarks. 
    The points can be directly on the cropped image.

    We can also adapth the points back to the actual image.

    I think we need to make the point based on the actual image so that croppint the eye images will be 
    based on the image and not the cropped face.
    """
    # original_image, _ = FaceBiometricsUtils.readImage(self.image_filename, False, False)
    # h, w, c = original_image.shape
    bbox = result[0]['box']

    # Split the bbox
    x_box, y_box, xw_box, hh_box = bbox

    landmark_ = np.asarray(np.zeros(landmarks.shape)) 
    for i, point in enumerate(landmarks):
        x = point[0] * xw_box + x_box
        y = point[1] * hh_box + y_box
        landmark_[i] = (x, y)
    return landmark_.astype(np.int32)

def preprocessLandmark(image_mean, image=None):
    
#     image = cv2.resize(image, (112, 112))
    print("Shape of image: {} and shape of image_mean {}".format(image.shape, image_mean.shape))
    processed_image = (image - image_mean) / 128
    processed_image = np.rollaxis(processed_image, 2, 0)[None, :, :, :].astype(np.float32)
    return processed_image

def getFaceLandmarks(onnx_face_runtime, image, result, save_image_landmark=False):
    """
    Get face region from a face image
    We need to manipulate the 68 landmark from an image point
    """
    # self.onnx_face_runtime.run()
    # print(type(self.image), self.image.shape)
    image_mean = np.array([127, 127, 127])
    processed_image = preprocessLandmark(image_mean, image)
    face_landmarks_prediction = onnx_face_runtime.run(None, {'input': processed_image})
    face_landmarks_prediction = list(map(lambda x: x.reshape(-1, 2), face_landmarks_prediction))
    face_landmarks_prediction = projectLandmarks(face_landmarks_prediction[0], result)

    if save_image_landmark:
        # THIS WILL SAVE THE IMAGE WITH LANDMARKS TO YOUR CURRENT DIRECTORY
        original_image = cv2.cvtColor(cv2.imread("Image1.png"), cv2.COLOR_BGR2RGB)
        image_with_landmark = showImageWithLandmark(original_image, face_landmarks_prediction)
        cv2.imwrite("face_with_landmark.png", image_with_landmark)
    return face_landmarks_prediction


def head_in_frame(result, img):
    x, y, w, h = result[0]['box']
    imgH, imgW, _ = img.shape
    if x < 0 or y < 0 or y+h > imgH or x+w > imgW:
        raise ValueError('Head not in Frame.')

def face_centered(onnx_face_runtime, img, result):
    img = cv2.resize(img, (112, 112))
    face_landmarks_prediction = getFaceLandmarks(onnx_face_runtime, img, result)
    
    right_eye_y = face_landmarks_prediction[36][1]
    left_eye_y = face_landmarks_prediction[45][1]

    # Check that both eyes are not different with specified Threshold
    if (right_eye_y - left_eye_y) > 15:
        raise ValueError(" face-tilted-right")
    elif(left_eye_y - right_eye_y) > 15:
        raise ValueError(" face-tilted-left")

def load_tflite(model_path = "model.tflite" ):

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate tensors
    interpreter.allocate_tensors()

    return interpreter, output_details

def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

# This is to expand the box to ensure that bbox dosent cut the face or its 
# beyond the image.
def expand_bbox(faces, img):
    face_boxes = []
    for face in faces:
        x, y, w, h = face['box']
        left = x 
        if left < 0:
            left = 0
        right = w
        if left + w > img.shape[1]:
            right = img.shape[1] - left

        top = y 
        if top < 0:
            top = 0
        
        bottom = h
        if top + h  > img.shape[0]:
            bottom = img.shape[0] - top
        
        face_boxes.append([left, top, right, bottom])
    
    return face_boxes

# Adds border to image
def add_border(img):
    img = cv2.resize(img,(224,224))
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_WRAP)
    
    return img
    
def preprocessing(img_path, detector, face_model_path="", onnx_face_runtime="", validate_face=False):

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img)
    if not result:
        img = add_border(img)
        result = detector.detect_faces(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0) 

    if validate_face:
        x, y, w, h = result[0]['box']
        new_img = img[y:y+h, x:x+w]
        head_in_frame(result, img)
        face_centered(onnx_face_runtime, new_img, result)


    result = expand_bbox(result, img)
    x,y,w,h = result[0]
    new_img = img[y:y+h, x:x+w]
    cv2.imshow('image', new_img)
    cv2.waitKey(0) 
    new_img = new_img.astype(np.float32) / 255.
    new_img = cv2.resize(new_img, (112, 112))
    new_img  = np.expand_dims(new_img, axis=0)
    fliped = hflip_batch(new_img)
    return new_img, fliped