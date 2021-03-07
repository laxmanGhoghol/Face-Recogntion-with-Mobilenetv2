import cv2
import numpy as np
import os,glob

datadir = "img"
kaalu = glob.glob(datadir + '/kaalu/*')
lakha = glob.glob(datadir + '/lakha/*')
riya = glob.glob(datadir + '/riya/*')
data = []
labels = []


for i in kaalu:   
    image=cv2.imread(i)
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in lakha:   
    image=cv2.imread(i)
    image=np.array(image)
    data.append(image)
    labels.append(1)
for i in riya:   
    image=cv2.imread(i)
    image=np.array(image)
    data.append(image)
    labels.append(2)

data = np.array(data)
labels = np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, ytrain, ytest = train_test_split(data, labels, test_size=0.2,random_state=42)

np.save('data', data)
np.save('labels', labels)