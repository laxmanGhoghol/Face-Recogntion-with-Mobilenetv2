import cv2
import numpy as np
import os

def face_ext(img):
    
    blob=cv2.dnn.blobFromImage(img,1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections=net.forward()
    (h, w) = img.shape[:2]
    face = None

    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]

        if confidence>0.5:

            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
            face = img[startY: endY+10, startX: endX+10]
            break
    
    return face

prototxtPath=os.path.sep.join(['deploy.prototxt'])
weightsPath=os.path.sep.join(['res10_300x300_ssd_iter_140000.caffemodel'])
net=cv2.dnn.readNet(prototxtPath,weightsPath)
cap = cv2.VideoCapture(0)
count = 0
name = ""
pdir = "img/"

while True:
    print("Enter User Name: ")
    name = input()
    if not name.isalpha():
        print("Please enter only alphabetic character for your name!")
    else:
        break

path = os.path.join(pdir + "test/", name)
os.mkdir(path)
path = os.path.join(pdir + "train/", name)
os.mkdir(path)


while(True):
    _, img = cap.read()
    face = face_ext(img)
    cv2.putText(img, str(count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Collecting face data", img)
    if face is None:
        continue
    count += 1
    if count <= 70:
        imgpath = "img/train/" + name + "/" + str(count) + ".jpg"
    elif count > 70:
        imgpath = "img/test/"+ name  + "/" + str(count) + ".jpg"

    cv2.imwrite(imgpath, face)
    

    if cv2.waitKey(1) == 13 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
print("########## Completed #############")
