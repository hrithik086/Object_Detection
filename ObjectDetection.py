import cv2 as cv
import matplotlib.pyplot as plt
import pyttsx3
import math

configFile="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozenModel="frozen_inference_graph.pb"
tts=pyttsx3.init()

model=cv.dnn_DetectionModel(frozenModel,configFile)  #we have loaded pretrained tensorflow model in the memory
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)     #this will help to convert an bgr image to rgb image

targetList=["person", "bicycle", "car", "truck", "train", "traffic light", "fire hydrant", "stop sign", ]

labels=[]
fileName='Labels.txt'
with open(fileName,'rt') as fopen:
    labels=fopen.read().rstrip().split('\n')      #copyint all files to the list named labels these are coco names


capture=cv.VideoCapture(1,cv.CAP_DSHOW)

text=""
success, frame=capture.read()
while success and cv.waitKey(1)==-1:
    #cv.imshow('video',frame)
    labelIndex, confidence, bbox=model.detect(frame,confThreshold=0.5)
    #print(labels[labelIndex[0]-1])
    if len(labelIndex)>=1 and labelIndex[0]<79 and labels[labelIndex[0]-1] in targetList:
        '''if(text!=labels[labelIndex[0]-1]):
            text=labels[labelIndex[0]-1]
            print(text," ",confidence," ",bbox)
            tts.say(text)
            tts.runAndWait()'''
        
        #new code
        for classIndex, conf, boxes in zip(labelIndex.flatten(),confidence.flatten(),bbox):
            if(classIndex<80):
                cv.rectangle(frame,boxes,(0,0,255),2)

                x1=boxes[0]
                y1=boxes[1]
                x2=boxes[2]
                y2=boxes[3]
                x1=x2-x1
                y1=y2-y1
                x1=x1*x1
                y1=y1*y1
                x1=x1+y1
                x1=math.sqrt(x1)
                #print(x1)

                if(text!=labels[labelIndex[0]-1] and confidence[0]>0.6 and x1>400):
                    text=labels[labelIndex[0]-1]
                    print(text," ",confidence," ",bbox)
                    tts.say(text)
                    tts.runAndWait()


        #new code

    else:
        if text!="clear":
            text="clear"
            tts.say(text)
            tts.runAndWait()
    cv.imshow('video',frame)
    success,frame= capture.read()