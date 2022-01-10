import glob
import shutil
import os
from random import randrange
from utils import *

# [age]_[gender]_[race]_[date&time].jpg
# [age] is an integer from 0 to 116, indicating the age
# [gender] is either 0 (male) or 1 (female)
# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
# [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

face_groups = glob.glob("./data/face_dataset/*/")
face_dataset = {}
for group in face_groups:
    face_dataset[group.split('\\')[-2]] = glob.glob(group + "*")

# ├── DST_01.jpg(png)
# └── DST_02.jpg(png)
# └──...
# └── SRC_01.jpg(png)
# └── SRC_02.jpg(png)
# └──...

extracted_faces = glob.glob("./data/extracted_faces/*/*")
for i, face in enumerate(extracted_faces):
    tmp = face.split('\\')[-1].split('_')
    if len(tmp) < 3:
        continue
    group = '_'.join(tmp[:3])
    to_swap = face_dataset[group][randrange(len(face_dataset[group]))]
    root = './data/swap_faces/'
    src = root + 'SRC_' + str(i).zfill(2) + '.jpg'
    dst = root + 'DST_' + str(i).zfill(2) + '.jpg'

    shutil.copy(face, src)
    shutil.copy(to_swap, dst)
