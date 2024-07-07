Training ArcFace from Scratch

Clone the repository and navigate to the project directory.

Install the required dependencies by running pip install -r requirements.txt.

Generating TFRecords
To generate TFRecords from your image dataset, use the following command:

python src/tf_generator.py --dataset_path <path_to_images_folder> --tfrecord_name <name_of_tfrecord_file>

Example:

python src/tf_generator.py --dataset_path /path/to/images --tfrecord_name output.tfrecord

This will generate a TFRecord file named output.tfrecord in the specified checkpoint path.

Note: Make sure to update the dataset_path argument to the path where your image dataset is located.

Usage

To use this script, follow these steps:

Clone the repository and navigate to the project directory.

Install the required dependencies by running pip install -r requirements.txt.

Prepare your dataset in the following structure:

dataset_path/

  |-- id_1/
  
  |    |-- image1.jpg
  
  |    |-- image2.jpg
  
  |    ...
  
  |-- id_2/
  
  |    |-- image1.jpg
  
  |    |-- image2.jpg
  
  |    ...
  
  ...
  
Run the script using the command above.

Arguments

dataset_path: Path to the image dataset.

tfrecord_name: Name of the output TFRecord file.

checkpoint_path: Path to the checkpoint folder (default: /opt/ml/checkpoints).



To train the ArcFace model from scratch, follow these steps:

Prepare your dataset in TFRecord format and specify the path to the dataset using the --train_tfrecord_path argument.

Run the training script using the following command:

python train.py --train_tfrecord_path <path_to_dataset> --model_dir <path_to_model_folder> --checkpoint_path <path_to_checkpoint_folder>

Example:

python train.py --train_tfrecord_path /path/to/dataset.tfrecord --model_dir /opt/ml/model --checkpoint_path /opt/ml/checkpoints

Making Inference on 2 Images

To make inference on two images after training, follow these steps:

Prepare the two images you want to compare.

Run the inference script using the following command:

python src/inference_with_two_images --image1_path <path_to_image1> --image2_path <path_to_image2> --model_dir <path_to_model_folder>

Example:

python src/inference_with_two_images --image1_path /path/to/image1.jpg --image2_path /path/to/image2.jpg --model_dir /opt/ml/model

This will output the similarity score between the two images.

Note: Make sure to update the model_dir argument to the path where your trained model is saved.
