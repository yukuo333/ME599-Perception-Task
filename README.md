# ME599-Perception-Task
Buyao Lyu, Dang Cong Bui, Huu Hieu Ta, Xiaoyu Li, Xincheng Cao, Yu Chun Kuo

This repositiory includes three separate methods of object classification: Resnet50, InceptionV3, and Yolov5. Each prediction method is done separately.

# Resnet50 and InceptionV3 Instructions
For Resnet50 and InceptionV3, your repository should look like this. 

```javascript
Resnet50
final_project\
  ├── resnet2.py\
  ├── trainval_IMG\
  │  ├── 0\
  │  ├── 1\
  │  ├── 2\
  └── label.csv\

InceptionV3
final_project\
  ├── inception.py\
  ├── trainval_IMG\
  │  ├── 0\
  │  ├── 1\
  │  ├── 2\
  └── label.csv\
```

To train and obtain the prediction label, simply run resnet2.py or inception.py. An output csv file "prediction_label.csv" will be created in the same folder.


# Yolov5 Instructions
## step1: Get the trained weights
### choice 1: Train the model from scratch
Given the training dataset is too large to upload here, we give a brief instruction on how to train the model. The key point is to construct a training set.
Note that every image contains only one vehicle in class 0, 1 or 2, according to yolov5, the training data set shall be constructed in this way:
the training set constains two sub directories: trainset/images/... and trainset/labels/..., note that the name of "trainset" can be changed but "images" and "labels" should not be changed.  
In trainset/images, it contains the 7573 images of .jpg format, which are just the images we downloaeded directlt; and in trainset/labels, it contains the corresponding 7573 .txt files with the same file name as the images, and in each .txt file, the label as well as the block position containing the vehicle is provided in this way:  
####### example.txt ########  
1 0.529258 0.446768 0.0323929 0.0285171   
###########################################  
Note here 1 stands for the label, 0.529258 is the block center's x-coordinate (normalized), 0.446768 is the block center's y-coordinates (normalized), 0.0323929 is the block's width (normalized), and 0.0285171 is the block's height (normalized). The block containing the vehicle is constructed based on the given bbox coordinates.   
we have provided the code for doing this task, which is under yolov5_method directory, and the name is **data_process_yolov5.py**.  
After the training set is done, it can then be feed into train.py to train the model.  
Before doing the training, one also needs to modify the file data/mydata.ymal, to change the directories there for the training set aligns with your own training set's directory.


### choice 2: Use our trained weight
We have run the **train.py** code for 70/300 epochs (given time limitation) and the weight file (larger than 25MB and cannot be uploaded here) can be found based on this link:  
[our weight in google drive](https://drive.google.com/drive/folders/1sO_2jmsFzSGNHhf5USEjXP7da1DUI7xm)  
Download it and unzip to the directory runs/detect/, so that the consequent code can be executed, note the name of the weight is called **best.pt**


## step 2: set up the test set  
This step is relatively easy, but be sure to update the code in **detect.py** to align with the directory of your own test set directory, i.e, change this line of code in detect.py:     
####### detect.py #######  
parser.add_argument('--source', type=str, default='../test_data/images', help='source')  # file/folder, 0 for webcam    
#########################  
so that the "default" parameters takes the relative location of the test set, and it is just a set of images itself, and in our case, it has 2631 images.


## step 3: conduct detection
This step is simply run the code **detect.py**, and it will generate the labeled images in runs/detect/exp*/ (* differes by different run times, varies from empty, 1, 2, and increasing), moreover, the corresponding lables will be generated in runs/detect/exp*/labels/, and they are 2631 .txt files containing the detected labels and the corresponding box location (takes almost the same format as the training labels, but this time, the confidence level is also given). Here is an example:  
####### example.txt ########  
1 0.529258 0.446768 0.0323929 0.0285171 0.93  
###########################################  
Note the last term 0.93 here means the confidence level for the detected object, and also note that there is not necessary to be only one object detected in a single image, we will take the one with the largest confidence level.

## step 4: generate .csv file
Run the file **runs/detect/gen_test_file.py**, it will automatically generate a .csv file called **prediction_1.csv** under the same directory. This .csv file will be used to generate the final file to submit together with another two .csv files (we introduced weights in three methods and combined them).  

# Combining the results
Up to now we have three separate .csv files generated using three different methods, we give each method a weight of 0.33. Then, for a single image, we need to combine the 3 results for testing: if two or three of the methods classify the image in the same category, we will then assign the image to that category; if three methods generates 3 results, we will assign the image randomly to the three categories (this is a rare case). Sometimes yolov5 cannot detect the vehicle inside the image, if this is the case, we will run the same idea on the other two methods.  
We have provided the code to do this task, and the name is **combine.py**. Just download it with three .csv files under the same directory, then it will generate a new .csv file.
 
