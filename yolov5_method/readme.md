# Steps of using this code:
## step1: Get the trained Weights
### choice 1: Train the model from scratch:
Given the training dataset is too large to upload here, we give a brief instruction on how to train the model. The key point is to construct a training set.
Note that every image contains only one vehicle in class 0, 1 or 2, according to yolov5, the training data set shall be constructed in this way:
the training set constains two sub directories: trainset/images/... and trainset/labels/..., note that the name of "trainset" can be changed but "images" and "labels" should not be changed.  
In trainset/images, it contains the 7573 images of .jpg format, which are just the images we downloaeded directlt; and in trainset/lables, it contains the corresponding 7573 .txt files with the same file name as the images, and in each .txt file, the label as well as the block position containing the vehicle is provided in this way:  

