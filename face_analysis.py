import glob
import shutil
import argparse
import numpy as np
import secrets
import cv2
from utils import *
from deepface import DeepFace
from insightface_func.face_detect_crop_multi import Face_detect_crop

models = {}
models['age'] = DeepFace.build_model('Age')
models['gender'] = DeepFace.build_model('Gender')
models['race'] = DeepFace.build_model('Race')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help='Path to the data directory containing aligned your face patches.',
        default='./data/extracted_faces/pic4/')
    args = parser.parse_args()
    return args


def analyze_face(img):
    obj = DeepFace.analyze(img_path=img,
                           actions=['age', 'gender', 'race'],
                           models=models,
                           enforce_detection=False)

    print(obj)

    return obj['gender'], obj['age'], obj['dominant_race']


def move_file(file, gender, age, race):
    gender = 0 if gender == "Man" else 1
    age_group = mapAgeToAgeGroup(age)
    race_group = mapRaceToRaceGroup(race)

    location_to_move_to = dir + str(age_group) + "_" + str(gender) + "_" + str(
        race_group) + "_" + secrets.token_hex(10) + ".jpg"

    shutil.move(file, location_to_move_to)
    return


args = parse_args()
dir = args.dir

face_pics = glob.glob(dir + "*")

matrix = np.zeros((len(face_pics), len(face_pics)))
fmap = {}

face_groups = []
for i, f in enumerate(face_pics):
    if f.split('\\')[-1] in fmap or f.split('/')[-1] in fmap:
        continue
    df = DeepFace.find(img_path=f,
                       db_path=dir,
                       enforce_detection=False,
                       model_name="Facenet")
    same_faces = df['identity']
    same_faces = [u.split('/')[-1] for u in same_faces]
    for face in same_faces:
        fmap[face] = 1
    face_groups.append(same_faces)

app = Face_detect_crop(
    name='antelope',
    root='./insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(320, 320), mode=None)

final_faces = []
for i, g in enumerate(face_groups):
    print("group " + str(i))
    best_f = None
    best_score = 0.0
    for f in g:

        img = cv2.imread(dir + f)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces, points = app.detect(img)

        print(f, len(faces))
        if len(faces) > 0:
            print(faces[0, 4])
            if faces[0, 4] > best_score:
                best_score = faces[0, 4]
                best_f = f
    # if best_score >= 0.8:
    final_faces.append(best_f)

for i, f in enumerate(final_faces):
    gender, age, race = analyze_face(dir + f)
    move_file(dir + f, gender, age, race)
