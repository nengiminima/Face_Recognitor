Training ArcFace from Scratch

To train the ArcFace model from scratch, follow these steps:
Clone the repository and navigate to the project directory.

Install the required dependencies by running pip install -r requirements.txt.
Prepare your dataset in TFRecord format and specify the path to the dataset using the --train_tfrecord_path argument.
Run the training script using the following command:
python (link unavailable) --train_tfrecord_path <path_to_dataset> --model_dir <path_to_model_folder> --checkpoint_path <path_to_checkpoint_folder>
Example:
Bash
python (link unavailable) --train_tfrecord_path /path/to/dataset.tfrecord --model_dir /opt/ml/model --checkpoint_path /opt/ml/checkpoints
Making Inference on 2 Images
To make inference on two images after training, follow these steps:
Prepare the two images you want to compare.
Run the inference script using the following command:
python (link unavailable) --image1_path <path_to_image1> --image2_path <path_to_image2> --model_dir <path_to_model_folder>
Example:
Bash
python (link unavailable) --image1_path /path/to/image1.jpg --image2_path /path/to/image2.jpg --model_dir /opt/ml/model
This will output the similarity score between the two images.
Note: Make sure to update the model_dir argument to the path where your trained model is saved.
