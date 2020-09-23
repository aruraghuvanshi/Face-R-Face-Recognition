# Face-R 
Face-R is a Face Recognition script developed in Python which can be used with PC WEBCAM feed, an IMAGE and even on the DJI Tello flying Drone video feed. In fact it can be used on any video stream whose video stream port is known.

All Option and configurations can be set up by making changes to the config.py script.
There is no need to make any changes in the facerecognition2.py as it is designed so that users don't confuse themselves during changing figures; config takes care of that.

"train_names" is a list that contains labels of the images you want to train to be recognised. Ensure the serial order on the train_names list remains the same as the order of the image files in the dataset directory.

# How to Use:
Webcam Mode:
1. Open Config.py and make changes to add your dataset path that will be used to train your images.
2. To use webcam feed, set the mode flag of Face-R to 'webcam' keep dronecam = False
3. Run the facerecognition2.py script and bring the people in front of the webcam to test. Rectangles and names will appear on the recognised faces in the video feed.

Image Mode:
1. Open Config.py and add path of the test image to the static folder on which the faces have to be recognised.
2. Set the mode flag of the Face-R to 'image'.
3. Run the facerecognition2.py script. An image will pop-up with rectangled names of recognised faces from the list of train_names in config.py.

Drone Mode:
If you have the DJI Ryze Tello drone, you can use the face recognizer.
1. Open Config.py and make and set mode to 'webcam' and dronecam to 'True.
2. Run the facerecognition2.py script. The drone feed will contain rectangles of the recognised faces once those people appear in front of the drone.
