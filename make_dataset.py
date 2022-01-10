import requests
import shutil
import cv2
import os
import glob
import secrets
import numpy as np
import time
from deepface import DeepFace
from utils import *
import time
from time import sleep
from insightface_func.face_detect_crop_multi import Face_detect_crop

# settings
url = "https://thispersondoesnotexist.com/image"

temp_file = "img.png"
times_to_run = 1000
seconds_to_sleep = 1
crop_size = 224
root = "data/fake_faces/"

models = {}
models['age'] = DeepFace.build_model('Age')
models['gender'] = DeepFace.build_model('Gender')
models['race'] = DeepFace.build_model('Race')


def download_face():
    response = requests.get(url, stream=True)
    with open(temp_file, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    return


def analyze_face(img):
    obj = DeepFace.analyze(img_path=img,
                           actions=['age', 'gender', 'race'],
                           models=models,
                           enforce_detection=False)

    return obj['gender'], obj['age'], obj['dominant_race']


def write_file(img, gender, age, race):
    gender = 0 if gender == "Man" else 1
    age_group = mapAgeToAgeGroup(age)
    race_group = mapRaceToRaceGroup(race)
    group = str(age_group) + "_" + str(gender) + "_" + str(race_group)
    os.makedirs(root + group, exist_ok=True)
    location_to_move_to = root + group + '/' + group + "_" + secrets.token_hex(
        10) + ".png"

    cv2.imwrite(location_to_move_to, img)

    #shutil.move(file, location_to_move_to)
    return


app = Face_detect_crop(
    name='antelope',
    root='./Face-Track-Detect-Extract/insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640), mode=None)

faces = glob.glob("./data/fake_original_faces/*.png")
failed = []
for face in faces:

    print(face)

    # download_face()
    # shutil.move(temp_file, "data/fake_original_faces/" + str(a) + ".png")

    img = cv2.imread(face)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detected_face = DeepFace.detectFace(img_path=temp_file,
    #                                     detector_backend='opencv')

    # if detected_face.sum() == 0.0:
    #     continue

    # print(detected_face)

    # detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)


    bbox, kps = app.detect(img_rgb)
    print(bbox)
    if len(bbox) == 0:
        failed.append(face)
        continue
    aligned, _ = app.get(img_rgb, crop_size, 1)

    output = cv2.cvtColor(aligned[0], cv2.COLOR_RGB2BGR)
    
    gender, age, race = analyze_face(face)
    # write_file(output * 255, gender, age, race)
    write_file(output, gender, age, race)

print(failed)

