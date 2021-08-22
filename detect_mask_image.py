import sys
import os
import cv2
import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


from yoloface_master.utils import *


print("[INFO] loading face mask detector model...")
maskNet = load_model("model_detector.model")      #mask detector model
#####################################################################

model_cfg_path = "yoloface_master/cfg/yolov3-face.cfg"
model_weights_path = "yoloface_master/model-weights/yolov3-wider_16000.weights"

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', model_cfg_path)
print('[i] The weights of model file: ', model_weights_path)
print('###########################################################\n')


# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)       #face detector model
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detect_mask(frame):
    wind_name = 'face detection using YOLOv3'
    #cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    #frame = cv2.resize(frame, (200,300))
    # Stop the program if reached end of video
    if frame is not None:
        print('[i] ==> Done processing!!!')
        #print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
        cv2.waitKey(1000)

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))                  #face detection

    # Remove the bounding boxes with low confidence
    faces1 = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    print('[i] ==> # detected faces: {}'.format(len(faces1)))
    print('#' * 60)
    mask_detections = []

    # initialize the set of information we'll displaying on the frame
    info = [
            ('number of faces detected', '{}'.format(len(faces1)))
    ]

    centroids = []
    for box in faces1:                          # detect mask for each face
        (startX, startY, endX, endY) = box
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(IMG_WIDTH - 1, endX), min(IMG_HEIGHT - 1, endY))

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        faces = []
        locs = []
        preds = []

        face = frame[startY:endY, startX:endX]
        if face.size != 0:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            preds = maskNet.predict(faces)                  #mask detection



        for (pred,loc) in zip(preds,locs):
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            #color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            accuracy = max(mask, withoutMask) * 100
            (startX, startY, endX, endY) = loc
            (cX,cY) = (((startX+endX)/2),((startY+endY)/2))
            md = (accuracy,loc,(cX,cY),label)
            mask_detections.append(md)
        # display the label and bounding box rectangle on the output
        # frame
    return mask_detections






"""img = cv2.imread('samples/outside_000001.jpg')
result = detect_mask(img)
print(result)"""
