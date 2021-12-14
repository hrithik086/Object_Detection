import cv2 as cv
import matplotlib.pyplot as plt
import pyttsx3

configFile="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozenModel="frozen_inference_graph.pb"
tts=pyttsx3.init()

model=cv.dnn_DetectionModel(frozenModel,configFile)  #we have loaded pretrained tensorflow model in the memory
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)     #this will help to convert an bgr image to rgb image

labels=[]
fileName='Labels.txt'
with open(fileName,'rt') as fopen:
    labels=fopen.read().rstrip().split('\n')         #copyint all files to the list named labels these are coco names


capture=cv.VideoCapture(0)

text=""
success, frame=capture.read()
while success and cv.waitKey(1)==-1:
    #cv.imshow('video',frame)
    labelIndex, confidence, bbox=model.detect(frame,confThreshold=0.5)
    if len(labelIndex)>=1 and labelIndex[0] in range(1,80):
        '''if(text!=labels[labelIndex[0]-1]):
            text=labels[labelIndex[0]-1]
            print(text," ",confidence," ",bbox)
            tts.say(text)
            tts.runAndWait()'''
        
        #new code
        for classIndex, conf, boxes in zip(labelIndex.flatten(),confidence.flatten(),bbox):
            if(classIndex<80):
                cv.rectangle(frame,boxes,(0,0,255),2)

                if(text!=labels[labelIndex[0]-1]):
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