# bupt_AQA
Scoring and Classification

## Requirements

* pytorch
* torchvision
* tqdm
* requests

## Usage
```
cd models
python relic2_model.py --path_to_images=图像名称(.jpg后缀)
如：python relic2_model.py --path_to_images=150.jpg

美学评分及分类存于同目录score.txt文件中
```

图片案例1
--------------------
* The aesthetic score is:6.73.
* It is a good photo.
<p align="center">
  <img src="https://github.com/BUPTAQA/bupt_AQA/blob/main/AVA/models/150.jpg" alt="test_photo" width="48%">
</p>

图片案例2
--------------------
* The aesthetic score is:4.72.
* It is not a good photo.
<p align="center">
  <img src="https://github.com/BUPTAQA/bupt_AQA/blob/main/AVA/models/3.jpg" alt="test_photo" width="48%">
</p>
