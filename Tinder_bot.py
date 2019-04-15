from random import randint
import cv2
import sys
import os
import numpy as np
import fb_auth_token
import tinder_api as api
import re
import json
import urllib.request
import keras.models
import datetime
from keras.preprocessing import image

num_requests = int(input('Input the number of request batches to swipe through (each batch is perhaps a couple of dozen people): '))

# Boolean to determine whether photos of swiped persons should be saved or not (to folder 'Tinder photos').
save_photos = True

# Tries to update location with given GPS coordinates (requires premium).
#api.update_location(10,8042723,106,6530501)

# Loads the model that is used for facial beauty prediction. 
# This model is a pretrained ResNet50 from Keras, further trained on the SCUT-FBP5500 dataset. 
FBP_model = keras.models.load_model('FBP_model.h5')

# Creates the 'Tinder photos' folder if we want to save photos and it does not already exist.
if (not "Tinder photos" in os.listdir(".")) and save_photos:
    os.mkdir("Tinder photos")
    #os.chdir("Tinder photos")
    
# Needed for image preprocessing later.
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

# Get Facebook authorization for Tinder (insert your own FB login info). 
# OBS: If ran too frequently, may trigger FB temporary suspension.

#fb_access_token = fb_auth_token.get_fb_access_token('lufterik2@gmail.com', 'testarenmassa')
#fb_user_id = fb_auth_token.get_fb_id(fb_access_token)

fb_access_token = 'EAAGm0PX4ZCpsBAFr8TCZAnuO6703hV3KfopRWvyqFwl14P0OEvdWzjwZCa3U1WZA86mTHVmxSiYijr2wTJ6zpzID9NZASM72QJBnVMflZA2E3Y4a8spXZBvs40eNe2GcsbnE4dG5g6Dium2BzmKnjr2d1ZAbsf0fGPZBpXrYN7aWZCLOgEkKVfZAqtaNpxXBZCHoSBhjBXaVSTdpxX8pfAvyqmYYfsWp8ns4poVQGetEV3nUwmtOjOTgUtKo'
fb_user_id = '101989090993974'

print('FB access token:')
print(fb_access_token)
print('FB user id:')
print(fb_user_id)

# Get Tinder authentication token.
#tinder_auth_token = api.get_auth_token(fb_access_token, fb_user_id)
print('Tinder authentitacion token:')
#print(tinder_auth_token)

# Define function to extract and preprocess face images from photos. Results in 350x350 pixel images.
def extract_faces(image):

    processed_images = []

    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(75,75),flags=0)

    for x,y,w,h in faces:
        sub_img=image[y-15:y+h+15,x-15:x+w+15]
        side = np.max(np.array([sub_img.shape[0],sub_img.shape[1]]))
        sub_image_padded = cv2.copyMakeBorder(sub_img,int(np.floor((side-sub_img.shape[1])/2)),int(np.ceil((side-sub_img.shape[1])/2)),int(np.floor((side-sub_img.shape[0])/2)),int(np.ceil((side-sub_img.shape[0])/2)),cv2.BORDER_CONSTANT)
        sub_image_resized = cv2.resize(src = sub_image_padded,dsize=(350,350))
        processed_images.append(sub_image_resized)
    return processed_images

# Get recommendations, analyze and swipe them.
person_id=''
swipes = []

for request_ix in range(num_requests):
    for key, value in api.get_recommendations().items():
        if(key == "results"):
            for person in value:
                liked = False
                ratings = [0]
                for key, value in person.items():
                    if(key == "_id"):
                        person_id = value
                        print('Person id: ', person_id)
                    if(key == "photos"):
                        photo_no = 0
                        for photo in value:
                            processedFiles = photo['processedFiles']
                            temp = processedFiles[0]
                            url = (temp['url'])
                            urllib.request.urlretrieve(url, str(person_id)+'_'+str(photo_no)+'.jpg')
                            im=cv2.imread(str(person_id)+'_'+str(photo_no)+'.jpg')
                            try:
                                processed_images = extract_faces(im)
                            except:
                                pass
                            i = 0
                            for face in processed_images:
                                cv2.imwrite(str(person_id)+'_'+str(photo_no)+'_'+str(i)+'.jpg',face)
                                img=image.load_img(str(person_id)+'_'+str(photo_no)+'_'+str(i)+'.jpg')
                                img=image.img_to_array(img)
                                pred = FBP_model.predict(img.reshape((1,) + img.shape))
                                ratings.append(pred[0][0])
                                i+=1
                            photo_no+=1
                max_rating = max(ratings)
                if max_rating>3:
                    print(api.like(person_id))
                    liked = True
                swipes.append((person_id,max_rating,liked))

print(swipes)

F = open('log.txt', 'a')

for item in swipes:
    F.write(str(item) + "\n")
    
F.close()