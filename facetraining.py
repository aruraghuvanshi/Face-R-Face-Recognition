import face_recognition as fr
import os, time
import glob
from config import datasetpath


def trainAndEncode(path=datasetpath, verbose=True):

    face_encodings = []
    allfiles = glob.glob(path + '\\*.jpg')
    print(f'\nCommencing \033[1;33mModel Training\033[0m, please wait...')
    time.sleep(1)

    for x in allfiles:

        if verbose: print(f'Training Model on {x}')
        image = fr.load_image_file(os.path.abspath(x))
        image_face_encoding = fr.face_encodings(image)[0]
        face_encodings.append(image_face_encoding)

    print(f'\nFace Training \033[1;32mSuccessful\033[0m.')
    if verbose: print(face_encodings)
    return face_encodings


