import pickle
import numpy as np
import sklearn.metrics as metrics
import cv2
from tqdm import tqdm 
import os
import sys
import importlib
module = importlib.import_module("1705045_train")
Convolution = getattr(module, "Convolution")
MaxPooling = getattr(module, "MaxPooling")
Flattening = getattr(module, "Flattening")
Dense = getattr(module, "Dense")
ReLU = getattr(module, "ReLU")
SoftMax = getattr(module, "SoftMax")
Lenet = getattr(module, "Lenet")
Lenet_modified = getattr(module, "Lenet_modified")



def read_test_files(folder):
    all_images=[]
    all_names=[]

    image_folder=folder
    for filename in tqdm(os.listdir(image_folder)):

        if filename.endswith(".csv"):
            continue

        img = cv2.imread(image_folder+'\\'+filename,0)
        img = cv2.resize(img,(28,28))

        img = np.array(img)
        img = 255-img
        img = img/255

        #dilate
        kernel = np.ones((5,5),np.uint8)
        img = cv2.dilate(img,kernel,iterations = 1)

        temp=np.zeros((1,28,28))
        temp[0,:,:]=img

        all_images.append(temp)
        all_names.append(filename)
    all_images=np.array(all_images)
    return all_images,all_names

if __name__ == '__main__':
    folder_name=sys.argv[1]

    model=pickle.load(open('1705045_model.pickle', 'rb'))
    all_images,all_names=read_test_files(folder_name)

    y_preds=model.test(all_images)

    # create csv file with all_names and y_preds
    path=folder_name+'\\1705045_prediction.csv'
    print(path)
    with open(folder_name+'\\1705045_prediction.csv', 'w') as f:
        f.write('FileName,Digit\n')
        
        for i in range(len(all_names)):
            f.write(all_names[i]+','+str(y_preds[i])+'\n')
        f.close()
