# -*- coding: utf-8 -*-
import os
import random
from glob import glob
import time
import cv2, threading
import numpy as np
from numpy import expand_dims
from time import gmtime, strftime



from threading import Thread, Lock

from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions
from PIL import Image

from multiprocessing import Process, Queue
from datetime import datetime


# Initialize global variables to be used by the classification thread
# and load up the network and save it as a tensorflow graph
input_w, input_h = 416, 416
class_threshold = 0.65

# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh

	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			#if(objectness.all() <= obj_thresh): continue
			if (objectness <= obj_thresh).all(): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
	image = load_img(filename, target_size=shape)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image 

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

def akdraw(image, v_boxes, v_labels, v_scores):
	for i in range(len(v_boxes)):
		label_str = ''
		box = v_boxes[i]
		label_str += "%s (%.3f)" % (v_labels[i], v_scores[i])
		cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
		cv2.putText(image, label_str , (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX,	1e-3 * image.shape[0], (0,255,0), 2)   

# load and prepare an image
def loadimg(filename):
	image = load_img(filename)
	image = img_to_array(image)
	return image

def preprocess_image_data(image):
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

def image_preporcess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
              
class VideoGetCamA:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    def __init__(self):
        self.stream = cv2.VideoCapture("rtsp://192.168.123.151:554/11") # Set to 1 for front camera
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


def make_prediction151(frame):
	#print("Predict151")
	img = Image.fromarray(frame)
	image = img.resize(size=(416, 416))
	image = preprocess_image_data(image)   
	yhat = model.predict(image)
	if yhat_queue151.empty():
		#print("Put yhat into the queue")
		yhat_queue151.put_nowait(yhat)


print('Loading network...')
#graph = tf.compat.v1.get_default_graph() 
#model = load_model('yolo-tiny.h5')
model = load_model('./weights/yolo.h5')
print('Network loaded successfully!')
#model._make_predict_function()
times = []
yhat_queue151 = Queue(maxsize=10000)

#Start the video capture loop
video_getterA = VideoGetCamA().start()



while True:
	timeA = time.time()
	# if video_getter.stopped:
	# 	video_getter.stop()
	# 	break
	

	drawframe1=video_getterA.frame
	image_h, image_w = drawframe1.shape[:2]

	if yhat_queue151.empty():
		make_prediction151(drawframe1)
		continue
	else:
		yhatPred = yhat_queue151.get(block=False, timeout=None)
		boxes = list()
		for i in range(len(yhatPred)):
			boxes += decode_netout(yhatPred[i][0], anchors[i], class_threshold, input_h, input_w)
		t1 = time.time()
		correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
		do_nms(boxes, 0.5)
		v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold) 
		akdraw(drawframe1, v_boxes, v_labels, v_scores)
		timeB = time.time()
		times.append(timeB-timeA)
		times = times[-20:]
		cv2.putText(drawframe1, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
		for i in range(len(v_boxes)):
			if v_labels[i] is not "":
				print("Predict: ", strftime("%Y-%m-%d-%H-%M-%S", gmtime()), v_labels[i], v_scores[i])
				cv2.imwrite('./auto-cap/capture-'+strftime("%Y-%m-%d-%H-%M-%S", gmtime())+'.jpg', drawframe1)
		imS = cv2.resize(drawframe1, (1800, 1000))
		cv2.imshow("predict", imS)
		#print("---------------------------")
		#print("")
		if cv2.waitKey(1) == ord("q"):
			break      
	
video_getterA.stop()
cv2.destroyAllWindows()