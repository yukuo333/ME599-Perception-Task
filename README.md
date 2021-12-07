# ME599-Perception-Task
Buyao Lyu, Dang Cong Bui, Huu Hieu Ta, Xiaoyu Li, Xincheng Cao, Yu Chun Kuo

This repositiory includes three separate methods of object classification: Resnet50, InceptionV3, and Yolov5. Each prediction method is done separately.

# Instructions
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
