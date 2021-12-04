# Steps of using this code:
## step1: Get the trained weights
### choice 1: Train the model from scratch
Given the training dataset is too large to upload here, we give a brief instruction on how to train the model. The key point is to construct a training set.
Note that every image contains only one vehicle in class 0, 1 or 2, according to yolov5, the training data set shall be constructed in this way:
the training set constains two sub directories: trainset/images/... and trainset/labels/..., note that the name of "trainset" can be changed but "images" and "labels" should not be changed.  
In trainset/images, it contains the 7573 images of .jpg format, which are just the images we downloaeded directlt; and in trainset/labels, it contains the corresponding 7573 .txt files with the same file name as the images, and in each .txt file, the label as well as the block position containing the vehicle is provided in this way:  
####### example.txt ########  
1 0.529258 0.446768 0.0323929 0.0285171   
###########################################  
Note here 1 stands for the label, 0.529258 is the block center's x-coordinate (normalized), 0.446768 is the block center's y-coordinates (normalized), 0.0323929 is the block's width (normalized), and 0.0285171 is the block's height (normalized). The block containing the vehicle is constructed based on the given bbox coordinates.   B 
After the training set is done, it can then be feed into train.py to train the model.  
Before doing the training, one also needs to modify the file data/mydata.ymal, to change the directories there for the training set aligns with your own training set's directory.


### choice 2: Use our trained weight
We have run the train.py code for 70/300 epochs (given time limitation) and the weight file (larger than 25MB and cannot be uploaded here) can be found based on this link:  
[our weight in google drive](https://drive.google.com/drive/folders/1sO_2jmsFzSGNHhf5USEjXP7da1DUI7xm)  
Download it and unzip to the directory runs/detect/, so that the consequent code can be executed, note the name of the weight is called **best.pt**


## step 2: set up the test set  
This step is relatively easy, but be sure to update the code in detect.py to align with the directory of your own test set directory, i.e, change this line of code in detect.py:  





