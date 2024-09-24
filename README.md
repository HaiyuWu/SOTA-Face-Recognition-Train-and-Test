
# SOTA-FR-train-and-test
This repository provides a neat package to efficiently train and test state-of-the-art face recognition models.

***Partial-FC training is supported for all the supported methods***.

### Supported Methods
- SphereFace (**CVPR17**) :white_check_mark:
- CosFace (**CVPR18**) :white_check_mark:
- ArcFace (**CVPR19**) :white_check_mark:
- ArcFace - combined margin (**CVPR19**) :white_check_mark:
- CurricularFace (**CVPR20**) :white_check_mark:
- MagFace (**CVPR21**) :white_check_mark:
- AdaFace (**CVPR22**) :white_check_mark:
- UniFace (**ICCV23**) :white_check_mark:
## Table of contents

<!--ts-->
- [Dataset preparation](#dataset-preparation)
  * [Hadrian and Eclipse](#hadrian-and-eclipse)
  * [Training sets](#training-sets)
  * [Test sets](#test-sets)
- [Train your own model](#train-your-own-model)
- [Test your own model](#test-your-own-model)
- [Model Zoo](#model-zoo)
- [Feature extraction](#feature-extraction)
- [Test SOTA models](#test-sota-models)
  * [Model Weights](#model-weights)
  * [Testing](#testing)
- [Reference](#reference)
- [Publications](#publications)
- [Acknowledgement](#acknowledgement)
- [License](#license)
  <!--te-->

## Environment
I suggest you to use Anaconda to better control the environments
```
conda create -n fr_training python=3.8
conda install -n fr_training pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
conda activate fr_training
```
Then clone the package and use pip to install the dependencies
```
git clone https://github.com/HaiyuWu/SOTA-FR-train-and-test.git
cd ./SOTA-FR-train-and-test
pip install -r requirements.txt
```

## Dataset preparation
### Hadrian and Eclipse
Hadrian and Eclipse are face recognition test sets oriented around facial hairstyles and face exposure levels, respectively. 
You can download both datasets via [GDrive](https://drive.google.com/file/d/1Q23Ze35jzzyR8k9h532gXdEpUJLC0SCR/view?usp=sharing). 
To get the password, you need to fill this [form](https://forms.gle/6Zuw9c6sdkU27hxN6), then you can follow [Test sets](#test-sets) to prepare the dataset.
If you find Hadrian and Eclipse help any of your projects, please cite the following reference:
```
@article{GoldilocksFRTestSet2024,
  title={What is a Goldilocks Face Verification Test Set?},
  author={Wu, Haiyu and Tian, Sicong and Bhatta, Aman and Gutierrez, Jacob and Bezold, Grace and Argueta, Genesis and Ricanek Jr., Karl and King, Michael C. and Bowyer, Kevin W.},
  year={2024}
}
```
The data used in the Hadrian and Eclipse datasets are fully based on the commercial version of [MORPH5](https://uncw.edu/myuncw/research/innovation-commercialization/technology-portfolio/morph).
We sincerely and heartfelt appreciate the invaluable support from [Prof. Karl Ricanek Jr.](https://people.uncw.edu/ricanekk/) and [University of North Carolina Wilmington](https://uncw.edu/) (UNCW).
UNCW has granted permission to use the images in Hadrian and Eclipse **FREE** for research purposes.
***You can get the full academic and commercial MORPH datasets at [official webpage](https://uncw.edu/myuncw/research/innovation-commercialization/technology-portfolio/morph)***.

If you have any problems using Hadrian and Eclipse, please contact: [cvrl@nd.edu](mailto:cvrl@nd.edu)
#### Convert to .bin file
If you just want to use the .bin version, using [xz2bin.py](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test/blob/main/utils/xz2bin.py) to convert. 
### Training sets
#### Option 1
You can directly download the compressed [MS1MV2](https://drive.google.com/file/d/10MaJjn3wvTcDCoXJdNmhMeAsRHfPuM-_/view?usp=drive_link)
, [WebFace4M](https://drive.google.com/file/d/12C9GvOEDcfqKm5XI5Ta2XvRBqlSy29C9/view?usp=drive_link), [Glint360K](https://drive.google.com/file/d/1WaLfIVJ7lQrwVgBOSp0BLNSUxLfFPccb/view?usp=drive_link).
Extract them at ***datasets*** folder and they are ready-to-use.

#### Option 2
All the other training sets could be found at [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).
After extracting the training set by using [rec2image.py](https://github.com/deepinsight/insightface/blob/0b5cab57b6011a587386bb14ac01ff2d74af1ff9/recognition/common/rec2image.py),
using [file_path_extractor.py](./file_path_extractor.py) to gather the image paths of the training set.
Then run [imagelist2lmdb.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/imagelist2lmdb.py) to finish the training set preparation.
```
python3 utils/imagelist2lmdb.py \
--image_list file/of/extracted/image/paths
--destination ./datasets
--file_name dataset/name
```

#### Option 3
We support using .txt file to train the model. Using [file_path_extractor.py](./file_path_extractor.py) to get all the image paths and replacing the path in config file.
```python
config.train_source = "path/to/.txt/file"
```
### Test sets
LFW, CFP-FP, CALFW, CPLFW, AgeDB-30 can be downloaded [here](https://drive.google.com/file/d/1l7XmqzIZKdKVqu0cOS2EI0bL_9_-wIrc/view?usp=drive_link).
Extract the compressed file then you can simply run [prepare_test_images.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/prepare_test_images.py) to get datasets ready to test
```
python3 utils/prepare_test_images.py \
--xz_folder folder/contains/xz/files
--destination ./test_set_package_5
--datasets lfw cfp_fp agedb_30 calfw cplfw
```

## Train your own model
After finishing the training and testing sets preparation, you train your own model by:
```
torchrun --nproc_per_node=4 train.py --config_file ./configs/arcface_r100.py
```
You can change the settings at [configs](https://github.com/HaiyuWu/SOTA-FR-train-and-test/tree/main/configs).

Note that, AdaFace and MagFace use BGR channel to train the model, but this framework consistently uses RGB to train the model. Also, for MagFace, it uses ```mean=0.``` and ```std=1.``` to normalize the images, but this framework uses ```mean=0.5``` and ```std=0.5``` to train and test all the methods.

If you want to train the model align with the training in the original GitHub repository, you can change the [data_loader_train_lmdb.py](./data/data_loader_train_lmdb.py) file.

## Test your own model
For CosFace, SphereFace, ArcFace, CurricularFace, UniFace, adding ```--add_flip``` option to test. For AdaFace, adding ```--add_norm``` option to test.
```
python3 test.py \
--model_path path/of/the/weights \
--model iresnet \
--depth 100 \
--val_list lfw cfp_fp agedb_30 calfw cplfw \
--val_source ./test_sets
```

## Model Zoo
The trained weights can be downloaded at [model zoo](./model_zoo)

## Feature extraction
Using [file_path_extractor.py](./file_path_extractor.py) to collect the paths of the target images, then run following command to extract the features.
```
python3 feature_extractor.py \
--model_path path/of/the/weights \
--model iresnet \
--depth 100 \
--image_paths image/path/file \
--destination feature/destination
```

## Test SOTA models
Now, we support testing for ArcFace (CVPR19), CurricularFace(CVPR20), MagFace(CVPR21), AdaFace(CVPR22), and TransFace(ICCV23).
### Model Weights
- Download [weights](https://github.com/deepinsight/insightface/tree/master/model_zoo#list-of-models-by-various-depth-iresnet-and-training-datasets) for ArcFace.
- Download [weights](https://github.com/HuangYG123/CurricularFace?tab=readme-ov-file#model) for CurricularFace
- Download [weights](https://github.com/IrvingMeng/MagFace?tab=readme-ov-file#model-zoo) for MagFace
- Download [weights](https://github.com/mk-minchul/AdaFace?tab=readme-ov-file#pretrained-models) for AdaFace
- Download [weights](https://github.com/DanJun6737/TransFace?tab=readme-ov-file#transface-pretrained-models) for TransFace
### Testing
Testing the pre-trained model (e.g. ArcFace-R100) on LFW, CFP-FP, CALFW, CPLFW, AgeDB-30 by
```
cd ./sota_test

python3 arcface_test.py \
--model_path path/of/pre-trained/model \
--net_mode ir \
--depth 100 \
--batch_size 512 \
--val_list lfw cfp_fp agedb_30 calfw cplfw \
--val_source ../test_set_package_5
```

### Acknowledgement
Thanks for the valuable contribution of [InsightFace](https://github.com/deepinsight/insightface/tree/master) in face area!

## Publications
[1] Identity Overlap Between Face Recognition Train/Test Data: Causing Optimistic Bias in Accuracy Measurement

[2] What is a Goldilocks Face Verification Test Set?

[3] Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors

## Reference
If you find this repo is helpful, please cite our paper.
```
@article{wu2024identity,
  title={Identity Overlap Between Face Recognition Train/Test Data: Causing Optimistic Bias in Accuracy Measurement},
  author={Wu, Haiyu and Tian, Sicong and Gutierrez, Jacob and Bhatta, Aman and {\"O}zt{\"u}rk, Ka{\u{g}}an and Bowyer, Kevin W},
  journal={arXiv preprint arXiv:2405.09403},
  year={2024}
}
```

## TODO list
Functions:
- [ ] resume from training

Methods:
- [ ] Circle loss

Others:
- [x] Published papers
- [x] references
## License
[MIT license](./license.md)
