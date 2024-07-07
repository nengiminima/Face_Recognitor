#import the necessary libraries 
import time
import boto3
import requests
import argparse
from smart_open import open
from selenium import webdriver
from urllib.error import HTTPError
from requests.exceptions import SSLError
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import  ReadTimeoutError
from requests.exceptions import ConnectionError
from botocore.exceptions import NoCredentialsError
from selenium.webdriver.chrome.options import Options

#command line argument parser
def parse_args():

    #Create the parser 
    parser = argparse.ArgumentParser(description="Script for web scraping", allow_abbrev=False)

    #Adding 2 arguments 
    parser.add_argument('access_key',
                        type = str,
                        help = 'This is your IAMUSER access key')

    parser.add_argument('secret_key',
                        type = str,
                        help = 'TThis is your IAMUSER secret key ' )

    #parser.add_argument('exe_path',
     #                   type = str,
      #                  help = 'This is the path to the executable driver for the browser!')

    #parser.add_argument('img_path',
    #                    type = str,
    #                    help = 'This is the path to the image directory!!')

    #parser.add_argument('namefile_path',
    #                    type = str,
     #                   help = 'This is the path to the text file containing names of images to scrape!!')


    args =parser.parse_args()

    return args

#Connection to s3 bucket 
def connect_to_s3():
    access_key = parse_args().access_key
    secret_key = parse_args().secret_key
    s3 = boto3.resource('s3',region_name='eu-west-1', aws_access_key_id=access_key, aws_secret_access_key= secret_key)
    s3client = boto3.client('s3', region_name='eu-west-1', aws_access_key_id=access_key, aws_secret_access_key= secret_key)
    return s3 , s3client

def image_scrape(bucket_path,log_path, log_list, link_log_list ):
    s3, s3client = connect_to_s3()

    #parse argument 
    #exe_path = parse_args().exe_path
    img_path = 'DeepLearning/Mobilefacenet/Dataset/Google_images' #parse_args().img_path
    namefile_path = 'DeepLearning/Mobilefacenet/src/finalList.txt'#parse_args().namefile_path

    #this options arguments are required for the sake of using headless chrome in EC2
    options = Options()
    options.add_argument("--headless")
    options.add_argument("window-size=1400,1500")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("enable-automation")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    browser = webdriver.Chrome(chrome_options=options)
    extensions = { "jpg", "jpeg", "png", "gif" }

    #Extracting each name from the text file and creat a list from it 
    obj = s3.Object(bucket_path, namefile_path)
    body = obj.get()['Body'].read()
    names = body.decode('utf-8')
    names = names.split('\n')

    #extracting logs from the log file saved in s3 and appending it into a list
    for line in open(log_path):
        log_list.append(line.strip().split('\t'))
    
    #extracting the links from each logs in the log file and appending it into a list 
    for _ in log_list:
        link_log_list.append(_[1])

    #This is a timer i used it to monitor the download time   
    start_time = time.time()

    #This checks for the info of the last image logged
    if len(log_list) == 0:
        index = 0 
    else:
        last_name = log_list[-1][0]
        last_link = log_list[-1][1]
        last_count = log_list[-1][2]
        index = names.index(last_name)
    new_log_list = log_list.copy()

    for searchterm in names[index:]:
        print(searchterm)
        #opens a search browser using a serach name 
        url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
        browser.get(url)

        #This helps to pick up from where it stopped using the last recorded info in the list ( This is done incase scripts stops running)
        if len(log_list) == 0:
            img_count = 0
            
        elif log_list[-1] and searchterm == last_name:
            img_count = int(last_count) + 1
        
        else:
            img_count = 0

        #This scrolls down the page 
        for _ in range(500):
            browser.execute_script("window.scrollBy(0,10000)") #Note there is an issue with this, i am still not able to scrape more than 100 images to be fixed later 
        
        #web scraping part of the script
        html = browser.page_source.split('["')
        for i in html:
            if i.startswith('http') and "https://encrypted" not in i:
                try:
                    #This extract the exact extension for the image

                    extension = '' 
                    for g in extensions:
                        if g in i.split('"')[0].split('.')[-1]:
                            extension = g

                    if extension == "":
                        continue
                        
                    #creating a name to store the downloaded image as in s3 
                    s3_image_filename = searchterm + str(img_count) + '.' + extension
            
                    
                    link = i.split('"')[0]

                    if link not in link_log_list:

                        #This downloads the content of the url 
                        req_for_image = requests.get(link,allow_redirects=False, stream=True)
                        file_object_from_req = req_for_image.raw
                        req_data = file_object_from_req.read()
                        
                        # This does the actual upload to s3
                        s3client.put_object(Bucket=bucket_path, Key=(img_path +'/' +searchterm + '/' + s3_image_filename), Body=req_data)
                        
                        #Logging new downloaded image data into list
                        new_log_list.append([searchterm, link, img_count])

                        end_time = time.time()
                        current_time = (end_time-start_time)
                        print(link)
                        print("done", s3_image_filename, extension)

                        #This is used to append to the current list to the image_log in s3 every 5 minutes
                        if current_time >= 30:
                            with open('/home/ubuntu/src/image_log.txt', 'w') as f:
                                for item in new_log_list:
                                    f.write(item[0] + '\t' + item[1] + '\t' + str(item[2]) + "\n")
                            s3.Object(bucket_path, "DeepLearning/Mobilefacenet/src/image_log.txt").delete()
                            s3client.upload_file("image_log.txt", bucket_path, "DeepLearning/Mobilefacenet/src/image_log.txt")
                            start_time = time.time()
                            print("done")
                        img_count += 1
                    #object = s3.Object(bucket_path, 'DeepLearning/Mobilefacenet/src/result.txt')
                    #object.put(Body=body)
                except (HTTPError, SSLError, MaxRetryError, ConnectionError, ReadTimeoutError):
                    pass


if __name__ == '__main__':
    #This  script is used to webscrape images from google and save them in s3 bucket (will run the script in ec2 instance )
    bucket_path = 'seamfix-machine-learning-ir'
    log_path = "s3://seamfix-machine-learning-ir/DeepLearning/Mobilefacenet/src/image_log.txt"
    log_list =  []
    link_log_list = []
    image_scrape(bucket_path, log_path, log_list, link_log_list )
    
