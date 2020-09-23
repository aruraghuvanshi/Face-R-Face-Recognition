'''

CONFIGURATIONS FILE

'''

import cv2

datasetpath = 'C:\\Users\\void_\\PycharmProjects\\FaceRecognition\\PROJECT FACEX\\dataset2'
imagepath = 'C:\\Users\\void_\\PycharmProjects\\FaceRecognition\\PROJECT FACEX\\static\\test3.jpg'


verbose = False                             # to see background computations and numeric video stream


mode = 'webcam'; dronecam = False           # modes are 'webcam', 'image'


im = cv2.imread(imagepath)
if im.shape[0] > 2000:
    scale = 0.25                            # For large jpgs like camera pictures from phones or high MP
elif 1000 > im.shape[0] >= 2000:
    scale = 0.5                             # For medium jpgs
elif 500 > im.shape[0] >= 1000:
    scale = 0.75                            # For whatsapp jpgs or small compressed images
else:
    scale = 1.0


trainflag = False                           # Turn True If a new model has to be trained else False

'''
The list train_names contains labels of the 
Faces that were encoded during training.
For each addition of new images to dataset2
the corresponding Name should be added to the 
end of the train_names list. All names in the 
dataset2 directory should be marked in a serial
order corresponding to the name in the list below.

'''

train_names = ['add', 'your', 'names', 'here']



