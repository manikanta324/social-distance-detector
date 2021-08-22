#import packages
#import cx_Oracle
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
from datetime import date,timedelta,datetime
from yoloface_master.detect_mask_image import detect_mask
#db connect
#con = cx_Oracle.connect('SYSTEM/ushapalani@127.0.0.1/XE')
#cur = con.cursor()
#cur.execute("""TRUNCATE TABLE social_distance""")

#Load coco class
labelsPath = os.path.sep.join([config.MODEL_PATH,'coco.names'])     #yolo model   config is social_distancing_config
LABELS = open(labelsPath).read().strip().split('\n')


#derive path for yolo weight and model config
weightsPath = os.path.sep.join([config.MODEL_PATH,'yolov3.weights'])
configPath = os.path.sep.join([config.MODEL_PATH,'yolov3.cfg'])

#load yolo object detector
#print('loading YOLO from disk...')
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

# check if we are going to use GPU
#if config.USE_GPU:
    # set CUDA as the preferable backend and target
    #print("[INFO] setting preferable backend and target to CUDA...")
#    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
 #   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#determine only the output layer names that need from yolo
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
names = ['trimmed-000-LONDON WALK _ Oxford Street to Carnaby Street _ En(480P).mp4','Egmore.mp4','Parryscorner.mp4','Pondy Bazaar.mp4','Triplicane.mp4']
titles = ['Chaepauk','Egmore','Parryscorner','Pondy Bazaar','Triplicane']
vs = [cv2.VideoCapture('videos/'+i) for i in names]
#loop over frames
initial = datetime.now()
data = []
#total_frames = 0
frame = [None]*len(names)
ret = [None]*len(names)
total_frames = [0]*len(names)
start = datetime.now()
x = date.today()
day = (x.strftime("%d")+'-'+x.strftime("%b").upper()+'-'+x.strftime("%y"))
flag = 0
while True:
    #
    for index,video in enumerate(vs):     #for each video

        total_frames[index] += 1
        ret[index],frame[index] = video.read()          #read a frame
        if total_frames[index]%10:
                continue
        if not ret[index]:
            flag = 1
            break

        #frame[index] = imutils.resize(frame[index],width = 700)

        if frame[index] is None:
            print('[i] ==> Done.......!!!')
        mask_detection = detect_mask(frame[index])          #mask detection

        results = detect_people(frame[index],net,ln,personIdx = LABELS.index('person'))       #person detection

        accuracy = [0]*len(results)
        label = ['None'] * len(results)
        new_label = []

        person_centroids = np.array([r[2] for r in results])
        face_centroids = np.array([r[2] for r in mask_detection])
        print(mask_detection)
        for (acc,box,(fcX,fcY),lab) in mask_detection:        #integrating
            ind = 0
            i = 0
            min = 10000
            for (pcX,pcY) in person_centroids:
                if(abs(pcX-fcX) < min):
                    ind = i
                    min = abs(pcX-fcX)
                i = i+1
            label[ind] = lab
            accuracy[ind] = acc



        for i in range(0,len(label)):               #assigning labels (both having mask -2, only one with mask - 1...)
            for j in range(i+1,len(label)):
                if(label[i] == 'Mask' and label[j] == 'Mask'):
                    new_label.append(2)
                elif((label[i] == 'Mask' and label[j] != 'Mask') or (label[i] != 'Mask' and label[j] == 'Mask')):
                    new_label.append(1)
                elif(label[i] == 'No Mask' and label[j] == 'No Mask'):
                    new_label.append(0)
                else:
                    new_label.append(-1)



        violate = set()
        index1 = 0
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids,metric = 'euclidean')
            for i in range(0,D.shape[0]):                       #assign min dis threshold based on label
                for j in range(i+1,D.shape[1]):
                    if(new_label[index1] == 2):
                        min_distance = config.MIN_DISTANCE2
                    elif(new_label[index1] == 1):
                        min_distance = config.MIN_DISTANCE1
                    elif(new_label[index1] == 0):
                        min_distance = config.MIN_DISTANCE0
                    else:
                        min_distance = config.MIN_DISTANCE
                    if D[i,j] < min_distance:
                        violate.add(i)
                        violate.add(j)
                    index1 = index1+1
        #Loop over results
        for (i, (prob, bbox, centroid)) in enumerate(results):          #assign color based on distance and mask.
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            acc = accuracy[i]
            lab = label[i]
            #color = (0,255,0)
            if i in violate:
                if(label[i] == 'Mask'):
                    color = (0,255,255)
                else:
                    color = (0, 0, 255)
            else:
                if(label[i] == 'Mask'):
                    color = (0,255,0)
                else:
                    color = (0,255,255)


            cv2.putText(frame[index],lab,(startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
            cv2.rectangle(frame[index], (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame[index], (cX, cY), 5, color, 1)
        text = 'Violations:{}'.format(len(violate))
        text1 = 'Total people in frame:{}'.format(len(results))
        end = datetime.now()
        fps = 0
        total_time =  abs(start.second - end.second)
        if total_time:
                fps = total_frames[index]/total_time
        #cv2.putText(frame[index],text,(10,frame[index].shape[0]-25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        #cv2.putText(frame[index],'fps:{:.2f}'.format(fps),(180,frame[index].shape[0]-25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        cv2.imshow(titles[index],frame[index])
        current = datetime.now()
        if int(current.second) - int(initial.second):
                timeset = str(current.hour)+":"+str(current.minute)+":"+str(current.second)
                title = titles[index]
                total = len(results)
                viol = len(violate)
                add = ("INSERT into social_distance_detector (Day, Place,Total_Count,Violated_Count) values(:1,:2,:3,:4)")
                data = [day,title,total,viol]
                #cur.execute(add,data)
                #con.commit()
                initial = current
    if flag:
        break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
#con.close()
