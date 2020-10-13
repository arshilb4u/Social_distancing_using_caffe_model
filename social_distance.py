from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time
from imutils.video import FPS
from math import pow, sqrt




print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe('models/SSD_MobileNet_prototxt.txt', 'models/SSD_MobileNet.caffemodel')

print('[Status] Starting Video Stream...')
vs = VideoStream(src = 0).start()
	
fps = FPS().start()

#Loop Video Stream
while True:

#Resize Frame to 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

#Converting Frame to Blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

#Passing Blob through network to detect and predict
	nn.setInput(blob)
	detections = nn.forward()

	F = 615
	pos = {}
	coordinates = {}

	for i in np.arange(0, detections.shape[2]):

		#Extracting the confidence of predictions
			confidence = detections[0, 0, i, 2]

		#Filtering out weak predictions
			if confidence > 0.5:

			#Extracting the index of the labels from the detection
				object_id = int(detections[0, 0, i, 1])

			#Identifying only Person as Detected Object
				if(object_id == 15):
					
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

				#Draw the prediction on the frame
					label = "Person: {:.2f}%".format(confidence * 100)
					cv2.rectangle(frame, (startX, startY), (endX, endY), (10,255,0), 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)

					coordinates[i] = (startX, startY, endX, endY)

				#Mid point of bounding box
					midOfX = round((startX+endX)/2,4)
					midOfY = round((startY+endY)/2,4)

					ht = round(endY-startY,4)

				 #Distance from camera based on triangle similarity
					distance = (F * 165)/ht
					
				 #Mid-point of bounding boxes (in cm) based on triangle similarity
					midOfX_cm = (midOfX * distance) / F
					midOfY_cm = (midOfY * distance) / F
					pos[i] = (midOfX_cm, midOfY_cm, distance)
					
	proximity = []


	for i in pos.keys():
			
			for j in pos.keys():

				if i < j:
					dist = sqrt(pow(pos[i][0] - pos[j][0],2) + pow(pos[i][1] - pos[j][1],2) + pow(pos[i][2] - pos[j][2],2))

				#Checking threshold distance - 175 cm
					if dist < 175:
						
						proximity.append(i)
						proximity.append(j)

						warning_label = "Maintain Safe Distance. Move away!"
						cv2.putText(frame, warning_label, (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
			
						
	for i in pos.keys():
		
		if i in proximity:

			color = [0,0,255]

		else:
			color = [0,255,0]
			
		(x, y, w, h) = coordinates[i]

		cv2.rectangle(frame, (x, y), (w, h), color, 2)

	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1) & 0xFF

	if key == ord('q'):
		break

cv2.destroyAllWindows()
vs.stop()
							
