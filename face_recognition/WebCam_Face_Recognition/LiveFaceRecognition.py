# import the necessary packages
from face_recognition import db,app
from face_recognition.WebCam_Face_Recognition import modules
from datetime import datetime
import numpy as np
from PIL import Image
from numpy import asarray
import imutils
import cv2
import traceback

def get_detections(caffe_model,frame):
	# image preprocessing like mean substraction to pass it in neural networks
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	# blob.shape = (1, 3, 300, 300)
	# pass the blob through the network and obtain the detections and
	# predictions
	caffe_model.setInput(blob)
	detections = caffe_model.forward()
	# detection.shape =  (1, 1, 200, 7)
	'''
    7 columns:
    3rd column = confidence of pizel
    4th column = (startX)/width
    5th column = (startY)/height
    6th column = (endX)/width
    7th column = (endY)/height
    '''

	return detections


def get_box(detection, dims):
	w = dims['w']
	h = dims['h']
	box = detection * np.array([w, h, w, h])
	return box.astype("int")


def draw_box(frame, dims):
	'''
    draw a box on face
    '''
	startX, startY, endX, endY = dims
	# draw the bounding box of the face along with the associated
	# probability
	cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)


def crop_image(frame, dims,required_size):
	'''
    returns face as a array
    '''
	startX, startY, endX, endY = dims
	# crop face from image array
	crop = frame[startY:endY, startX:endX]
	# convert array into image
	crop_img = Image.fromarray(crop)
	# resize face to (160,160,3)
	crop_img = crop_img.resize(required_size)
	# convert face image into array
	face = asarray(crop_img)
	return face


def predict_label(embs, svm_model,label_encoder):
	'''
    returns label name & prediction probability using (128,1) embeddings
    '''
	# to normalize face pixels
	encoder = modules.Normalizer()
	embs = encoder.transform(modules.expand_dims(embs, axis=0))

	# predict class lebel in numeric data type
	class_val = svm_model.predict(embs)
	# convert numeric label into string i.e. 0 -> karm
	class_name = label_encoder.inverse_transform(class_val)[0]
	# confidence to recognize
	class_probability = max(svm_model.predict_proba(embs)[0])

	return (class_name, class_probability)


def write_text(frame, data, dims,h):
	'''
    write label name on face
    '''
	startX, startY, endX, endY= dims
	text = str(data["class_name"]) + "-" + str(round(data["class_probability"] * 100, 2)) + "%"
	text_face = "face = " + str(round(data["detection_conf"], 2))
	# put text above rectangle
	# fix y position,where we have to put text
	y = startY - 10 if startY - 10 > 10 else endY + 10
	y_f = endY + 10 if endY + 10 < h else startY - 10
	cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
	cv2.putText(frame, text_face, (startX, y_f), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

def camera():

	#for face detection
	confidence_threshold = 0.5

	#for face recognition
	prediction_threshold = 0.5

	#for frame count to decide attendance
	frame_count_threshold = 5
	required_size = (160, 160)

	#get models
	facenet_model = app.config['FACENET_MODEL']
	caffe_model = app.config['CAFFE_MODEL']
	svm_model = app.config['SVM_MODEL']
	label_encoder = app.config['LABEL_ENCODER']

	# loop over the frames from the video stream
	print("starting video stream...")
	# capture from PC's Camera
	#vs = VideoStream(src=0).start()

	# Capture from Phone's Camera
	url = "http://192.168.0.103:8080/video"
	vs = cv2.VideoCapture(url)
	if not vs:
		print("camera can not capture frame!")

	# loop over the frames from the video stream
	frame_count = 0
	attendance_marked = {}
	attendance_started = {}
	while True:
		frame_count = (frame_count + 1) % 1000
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels

		#for mobile camera
		h, frame = vs.read()
		frame = imutils.resize(frame, width=360)
		#for pc camera
		#frame = vs.read()
		#frame = imutils.resize(frame, width=720)

		# grab the frame dimensions and convert it to a blob		
		(h, w) = frame.shape[:2]
		detections = get_detections(caffe_model,frame)
		# loop over the detections
		# for each in detections:
		for i in range(0, detections.shape[2]):

			# extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[0, 0, i, 2]
			# confidence = each['confidence']

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence

			if confidence < confidence_threshold:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			dims = get_box(detections[0, 0, i, 3:7], {"h": h, "w": w})
			startX, startY, endX, endY = dims

			# draw rectangle on face
			draw_box(frame, dims)

			try:
				# crop face from image
				face = crop_image(frame, dims,required_size)

				# get embeddings from image
				embs = modules.get_embeddings(facenet_model, face)

				# predict label
				class_name, class_probability = predict_label(embs, svm_model,label_encoder)

				if class_probability < prediction_threshold:
					class_name = "unknown"

				if class_name not in attendance_marked and class_name != "unknown":
					try:
						attendance_started[class_name]["count"] += 1
						attendance_started[class_name]["confidence"] += class_probability
					except:
						print(class_name, " recognized!")
						attendance_started[class_name] = {"count": 1, "confidence": class_probability}
					if attendance_started[class_name]["count"] >= frame_count_threshold:
						print(class_name, " Attendance marked!")
						true_count = attendance_started[class_name]["count"]

						attendance_marked[class_name] = {"frame_count":true_count,
														 "confidence":round(attendance_started[class_name]["confidence"] / true_count, 2),
														 "time":str(datetime.today().time())}
				# print(attendance_started)
				# print(frame_count)
				data = {"class_name":class_name,"class_probability":class_probability,"detection_conf":confidence}

				#write the label of person
				write_text(frame, data, dims,h)
			except Exception:
				traceback.print_exc()
				# print("title outside")
				pass

		#print(frame_count, end='\r')

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()

	# for pc camera
	#vs.stop()
	print(attendance_started)
	print(attendance_marked)
	# & 0xFF
	#for mobile camera
	vs.release()
	print("camera stopped")
	return attendance_marked
	#& 0xFF
