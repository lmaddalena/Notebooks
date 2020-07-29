import numpy as np
import h5py
import cv2
import os

# face mask dataset provided by Prajna Bhandary
# (https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)


def create_dataset():
    base_path = 'data'
    image_size = 128
    labels = [0,1]

    X = []
    Y = []

    for label in labels:
        if label == 0:
            image_path = os.path.join(base_path, 'without_mask')
        else:
            image_path = os.path.join(base_path, 'with_mask')

        # list files in the folder
        image_list = os.listdir(image_path)


        for img_file in image_list:
            # read the image
            img = cv2.imread(os.path.join(image_path, img_file))

            # convert the image to gray scale
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # resize the image
            img = cv2.resize(img, (image_size, image_size))

            # convert in numpy array
            img_arr = np.asarray(img)
            
            X.append(img_arr)
            Y.append(label)
            
    X_data = np.array(X)
    Y_data = np.array(Y)

    # create  h5 file
    hf = h5py.File(os.path.join(base_path, 'mask_data.h5'), 'w')
    hf.create_dataset('X_data', data=X_data)
    hf.create_dataset('Y_data', data=Y_data)

create_dataset()