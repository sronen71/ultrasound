import shutil
import requests

login_url = 'https://www.kaggle.com/account/login'
#download_url = 'https://www.kaggle.com/c/ultrasound-nerve-segmentation/download/train_masks.csv.zip'
download_url = 'https://www.kaggle.com/c/ultrasound-nerve-segmentation/download/train.zip'
#download_url = 'https://www.kaggle.com/c/ultrasound-nerve-segmentation/download/test.zip'




filename = download_url.split('/')[-1]
login_data = {'UserName':'sronen71', 
              'Password':'argo99'}

with requests.session() as s, open(filename, 'w') as f:
    s.post(login_url, data=login_data)                  # login
    response = s.get(download_url, stream=True)         # send download request
    shutil.copyfileobj(response.raw, f)                 # save response to file
