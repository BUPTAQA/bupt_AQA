# bupt_AQA
Scoring and Classification

## Requirements

* pytorch
* torchvision
* tqdm
* requests

## Usage

**Train model**
```
cd models
python relic2_model.py --path_to_images=图像名称(.jpg后缀)
如：python relic2_model.py --path_to_images=150.jpg
```
美学评分及分类存于同目录score.txt文件中：
* The aesthetic score is:6.73.
* It is a good photo.
