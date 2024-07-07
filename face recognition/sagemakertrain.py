import boto3
import sagemaker
from src.utils import load_yaml
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import TensorBoardOutputConfig


cfg = load_yaml('./config/arc_mbv2.yaml')

sess = boto3.Session()

bucket_name = cfg['bucket_name'] 
train_tfrecord_file = cfg['train_tfrecord_file']  
sagemaker_session = sagemaker.Session(boto_session=sess, 
                                        default_bucket = bucket_name)

def generate_data_location(bucket_name, prefix):
    
    return f"s3://{bucket_name}/{prefix}"

def get_role(): 
    """Get role for sagemaker execution."""
    return role

#Setting estimator parameters
use_spot_instances = False
max_wait = 432000 if use_spot_instances else None
dependencies = [ 'config', 'src', 'train.py']
output_path = cfg['output_path'] 


distributions = {'parameter_server': {
                    'enabled': True}}

# Use this when parsing bash scripts
tf_estimator = TensorFlow(entry_point       = 'train.sh',
                          dependencies      = dependencies,
                          instance_type     = 'ml.p2.xlarge',
                          output_path       = f'{output_path}/Model/',
                          checkpoint_s3_uri = f'{output_path}/Checkpoints',
                          role              = get_role(),
                          framework_version = '2.2.0',
                          instance_count    = 1,
                          py_version        = 'py37',
                          use_spot_instances= use_spot_instances,
                          max_run           = 432000,
                          max_wait          = max_wait,
                          sagemaker_session = sagemaker_session,
                          distribution      = distributions,
                          volume_size       = 70,
                          #script_mode=True,
                          )
                          
inputs = {'train': generate_data_location( bucket_name,train_tfrecord_file)}
tf_estimator.fit(inputs)
