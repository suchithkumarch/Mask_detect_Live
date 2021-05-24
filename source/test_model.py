'''
Packages used for the model
'''
import time
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

import os
from keras_preprocessing.image import img_to_array



def preprocess_face_frame(face_frame):
    # convert to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    # preprocess input image for mobilenet
    face_frame_resized = cv2.resize(face_frame, (224, 224))
    face_frame_array = img_to_array(face_frame_resized)
    return face_frame_array


def decode_prediction(pred):
    (mask, no_mask) = pred
    mask_or_not = "Mask" if mask > no_mask else "No mask"
    confidence = f"{(max(mask, no_mask) * 100):.2f}"
    return mask_or_not, confidence


def write_bb(mask_or_not, confidence, box, frame):
    (x, y, w, h) = box
    color = (0, 255, 0) if mask_or_not == "Mask" else (0, 0, 255)
    label = f"{mask_or_not}: {confidence}%"

    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def load_cascade_detector():
    cascade_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    return face_detector

mask_detection_model_path = 'models/final_mask_detection.h5'
default_face_path = 'models/haarcascade_frontalface_default.xml'
frontal_face_alt2_path = 'models/frontal_face_alt2.xml'
nose_path = 'models/Nariz.xml'
mouth_path = 'models/Mouth.xml'
eyes_path = 'models/frontalEyes35x16.xml'
#yolo_cfg_path = 'models/yolov4.cfg'
#yolo_weights_path = 'models/yolov4.weights'

#Classes that can be showed. 
class_names= ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
              'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
              'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
              'handbag tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
              'baseball','bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
              'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
              'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
              'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
              'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
              'teddy bear','hair drier','toothbrush']
              
bounding_colors = np.random.uniform(0,255,size = (len(class_names),3))

labels_dict={0:'MASK',1:'NO MASK',2:'WEAR MASK PROPERLY'}
color_dict={0:(0,255,0),1:(0,0,255),2:(0,100,100)}

model = keras.models.load_model(mask_detection_model_path)
face_detector = load_cascade_detector()

#network = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)  
# Using pretrained yolov4 model to detect the objects

face_clsfr=cv2.CascadeClassifier(default_face_path)
nose_clsfr = cv2.CascadeClassifier(nose_path)
mouth_clsfr = cv2.CascadeClassifier(mouth_path)
eyes_clsfr = cv2.CascadeClassifier(eyes_path)
frames = []

size_fact = (320,320)
mean = (0,0,0)
scalefactor = 0.004
yolo_predict = []

def output_layers(net):
    layer_names = net.getLayerNames()
    layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return layers
"""
def yolov4_detections(features,width,height,prob_thresh):
    ids,object_probability,bounding_box= [],[],[]
    for feature in features:
        for obj_prob in feature:
            scores = obj_prob[5:]
            id = np.argmax(scores)
            probability = scores[id]
            if(probability > prob_thresh):
                w,h = int(obj_prob[2]*width),int(obj_prob[3]*height)
                x,y = int(obj_prob[0]*width) - w / 2,int(obj_prob[1]*height) - h / 2
                ids.append(id)
                object_probability.append(float(probability))
                bounding_box.append([x,y,w,h])
    return bounding_box,object_probability,ids
""" 
def get_label(img):
    if(True):
        if(True):
            faces = face_clsfr.detectMultiScale(img,1.3,4)
            for (x,y,w,h) in faces:
                face_img = img[y:y+h,x:x+w]
                resized = cv2.resize(face_img,(224,224))
                normalized = resized/255.0
                reshaped = np.reshape(normalized,(1,224,224,3))
                result = model.predict(reshaped)
                label = np.argmax(result,axis=1)[0]
                accuracy = "{:.2f}".format(np.max(result) * 100)
                return [label,float(accuracy)]
#    return 2
"""    resized = cv2.resize(img,(224,224))
    normalized = resized/255.0
    reshaped = np.reshape(normalized,(1,224,224,3))
    result = model.predict(reshaped)
    label = np.argmax(result,axis=1)[0]
    accuracy = "{:.2f}".format(np.max(result) * 100)
    return [label,float(accuracy)]"""


def fun (i):
	dict={0:'MASK',1:'NO MASK'}
	return dict[i]



import unittest
import warnings 
import logging
logging.basicConfig(filename='mask_logs.log',format='%(asctime)s %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)
#logging.info('This is an info message')
def get_image(i):       
	path = "./sample_test_images/"+str(i)+".png"
	return cv2.imread(path)      
print("Beginning Testing...")

class TestModel(unittest.TestCase):
    def test1(self):        
        result,r = get_label(get_image(1))
        print("Label of Img ",1,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 1:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 0)
        self.assertNotEqual(result, 1)
      
    def test2(self):        
        result,r = get_label(get_image(2))
        print("Label of Img ",2,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 2:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 0)
        self.assertNotEqual(result, 1)
      
    def test3(self):        
        result,r = get_label(get_image(3))
        print("Label of Img ",3,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 3:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)

      
    def test4(self):        
        result,r = get_label(get_image(4))
        print("Label of Img ",4,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 4:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)
      
    def test5(self):        
        result,r = get_label(get_image(5))
        print("Label of Img ",5,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 5:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)
              
    def test6(self):        
        result,r = get_label(get_image(6))
        print("Label of Img ",6,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 6:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)
      
    def test7(self):        
        result,r = get_label(get_image(7))
        print("Label of Img ",7,":- ",result,"with accuracy",r,", i.e Status : ",fun(result))
        logger.info("Label of Img 7:- %d with accuracy %f i.e Status : %s",result,r,fun(result))
        self.assertEqual(result, 0)
        self.assertNotEqual(result, 1)
      

        
     

      


if __name__ == '__main__':
    unittest.main()
    
