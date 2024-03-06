import numpy as np
import cv2
import pickle
import sklearn


# Load all the tools required to be used 
haar_clf = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")
model = pickle.load(open("./model/svc_model.pickle", mode="rb"))
pca = pickle.load(open("./model/pca_dict.pickle", mode="rb"))
model_pca = pca["pca"]
mean_face = pca["mean_face"]
RESOLUTION = 100 # to fix the dimensions

def facerec_pipeline(filename: str, path: bool = True):
    # Load and preprocess
    if path:
        img = cv2.imread(filename)
    else: 
        img = filename
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get the FACE BBOXES
    faces_list = haar_clf.detectMultiScale(gray, 
                                       scaleFactor=1.5, 
                                       minNeighbors=3)
    results = [] # to collect the results of every single img from preprocessing stage
    # iterate the coordinates
    for x, y, width, height in faces_list:
        # draw a BBOX over original image
        cropped = gray[y: y + height, x: x + width]
        # normaliztion
        cropped = cropped / 255.
        # shrinkage
        if cropped.shape[1] > RESOLUTION:
            resized = cv2.resize(cropped, 
                                 (RESOLUTION, RESOLUTION), 
                                 cv2.INTER_AREA)
        # enlarging
        else: 
            resized = cv2.resize(cropped, 
                                 (RESOLUTION, RESOLUTION), 
                                 cv2.INTER_CUBIC)
        # reshaping 
        flat = resized.reshape(1, RESOLUTION ** 2)
        # subtract the pretrained MEAN FACE (already flatten)
        mean_flat = flat - mean_face
        # create an eigen image (apply the PCA n_comp = 60 by default)
        eigen = model_pca.transform(mean_flat)
        inverse = model_pca.inverse_transform(eigen)
        # predict gender
        output = model.predict(eigen)
        output_proba = model.predict_proba(eigen)
        confidence = output_proba.max()
        results_dict = {
            "cropped": cropped, 
            "eigen_image": inverse, 
            "gender": output[0], 
            "score": confidence
        }
        # accumulate the information
        results.append(results_dict)
        # attach the results with report
        report = "%s : %d" % (output[0], confidence * 100) 
        # diffentiate bbox colors by genders
        if output[0] == "female":
            color = (255, 0, 255)
        else:
            color = (255, 255, 0)
        # bbox for cropped face
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
        # bbox for report
        cv2.rectangle(img, (x, y - 30), (x + width, y), color, -1)
        cv2.putText(img, report, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2) 
    return img, results