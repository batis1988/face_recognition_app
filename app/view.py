from flask import render_template, request
import os
import cv2
from app.facerec import facerec_pipeline
import matplotlib.image as matimg

UPLOAD_FOLDER = "static/upload"



def index():
    return render_template("index.html")

def app():
    return render_template("app.html")

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path) # save image into upload folder
        pred_image, predictions = facerec_pipeline(path)
        pred_filename = "predicted_image.jpg"
        cv2.imwrite(f"./static/predict/{pred_filename}", pred_image)
        # reporting
        report = []
        for i, obj in enumerate(predictions):
            gray_img = obj["cropped"]
            eigen_img = obj["eigen_image"].reshape(100, 100)
            gender = obj["gender"]
            score = round(obj["score"] * 100, 2)
            gray_img_name = f"crop_{i}.jpg"
            eig_img_name = f"eig_{i}.jpg"
            matimg.imsave(f"./static/predict/{gray_img_name}", gray_img, cmap="gray")
            matimg.imsave(f"./static/predict/{eig_img_name}", eigen_img, cmap="gray")
            report.append([gray_img_name, eig_img_name, gender, score])
        
        return render_template("gender.html", fileupload=True, report=report) # post request
        
    return render_template("gender.html", fileupload=False) # get request

