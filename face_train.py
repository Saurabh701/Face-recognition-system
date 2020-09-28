import cv2
import numpy as np

#init the camera
cap = cv2.VideoCapture(0)


#Face detection
skip = 0
face_data = []
path = './ran/'
file_name = input('Enter the name of user data')
 
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()

	if(ret == False):
		continue
	
	faces = face_cas.detectMultiScale(frame,1.3,5)
	faces =sorted(faces, key = lambda  f:f[2]*f[3])
    #pick the last face
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		#extract region of intrest
		offset = 10
		face_sec = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_sec = cv2.resize(face_sec,(100,100))
		cv2.imshow("frame_pic",face_sec)
		skip+=1
		if(skip%10==0):
			face_data.append(face_sec)
			print(len(face_data))







	cv2.imshow("Frame",frame)
	

	#store every 10th face

	


	
	


	key_pres = cv2.waitKey(1) & 0xFF
	if key_pres == ord('Q'):
		break
#here we are saving our image into the numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save the image in file system

np.save(path+file_name+'.npy',face_data)

print("data saved")

cap.release()
cap.destroyAllWindows()	
