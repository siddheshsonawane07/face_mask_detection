# face-mask-recognition
For executing this project run the "detect.py" file after changing the directory of the files in "mask_detector.py" and "detect.py" file.

Dataset link: https://www.kaggle.com/omkargurav/face-mask-dataset

For mask_detector.py file:
When this file is executed, a "mask_detector.model" file is created. This model is then used by "detect.py" file for the loading of the model. Also, the pre-processing, graph of the training accuracy is written in this file.
Note: Change the "DIRECTORY" and "CATEGORIES" section of the file according to your directory. You can change "lr", "decay", "EPOCHS" values according to your need.

For detect.py file:
The model is executed with this file. Here, I have used the readymade face detector model downloaded from kaggle and use the "mask_detector" which we developed. 



