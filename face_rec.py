# read video stream using opencv
# extract faces out of it
# load ll the data (all numpy array)
     #x-values for faces
     #y-values for name
# we will use knn to find the pred
# we will map the predicted id to the name of pepole
# we will be dispalying the output display the name on the screen 


import cv2
import numpy as np
import os

############    KNN IMPLEMENTATION  #######
def dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
def knn(train,q,k=5):
    vals = [] #store the distances
    m = train.shape[0]
    for i in range(m):
    	ix = train[i,:-1]
    	iy = train[i,-1]
    	d = dist(q,ix)
    	vals.append((d,iy))
    vals = sorted(vals ,key=lambda x: x[0])
    vals = vals[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:,-1],return_counts = True)
    freq = np.argmax(new_vals[1])
    pred = new_vals[0][freq]
    return pred
###########################################
#init the camera
cap = cv2.VideoCapture(0)


#Face detection
skip = 0
path = './ran/'
face_data = []
labels  = []
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
class_id = 0 #labels for the face
names = {}  #mapping between id and the name


#Data preparation

for fx in os.listdir(path = './ran/'):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(path+fx)
		face_data.append(data_item)

		#create labels for class
		target = class_id*np.ones((data_item.shape[0]))
		class_id+=1
		labels.append(target)

face_dataset =  np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape(-1,1)
trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


#TESTING 

while True:
	ret,frame = cap.read()
	if(ret == False):
		continue
	faces = face_cas.detectMultiScale(frame,1.3,5)
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		#extract region of intrest
		offset = 10
		face_sec = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_sec = cv2.resize(face_sec,(100,100))

		#predicted label 
		out = knn(trainset,face_sec.flatten())

		#display the name and rectangle around it
		pred_name = names[int(out)]

		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow("Faces",frame)
	key_pres = cv2.waitKey(1) & 0xFF
	if key_pres == ord('q'):
		break

#here we are saving our image into the numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)







