
# SOTA-FR-train-and-test
This repository provides a neat package to efficiently train and test state-of-the-art face recognition models. 

## Table of contents

<!--ts-->
- [Dataset preparation](#dataset-preparation)
  * [Training sets](#training-sets)
  * [Test sets](#test-sets)
- [Train your own model](#train-your-own-model)
- [Test your own model](#test-your-own-model)
- [Feature extraction](#feature-extraction)
- [Test SOTA models](#test-sota-models)
  * [Model Weights](#model-weights)
  * [Testing](#testing)
- [Acknowledgement](#acknowledgement)
- [License](#license)
  <!--te-->

## Dataset preparation
### Training sets
#### Option 1
You can directly download the compressed [MS1MV2](https://drive.google.com/file/d/10MaJjn3wvTcDCoXJdNmhMeAsRHfPuM-_/view?usp=drive_link)
, [WebFace4M](https://drive.google.com/file/d/12C9GvOEDcfqKm5XI5Ta2XvRBqlSy29C9/view?usp=drive_link), [Glint360K](https://drive.google.com/file/d/1WaLfIVJ7lQrwVgBOSp0BLNSUxLfFPccb/view?usp=drive_link).
Extract them at ***datasets*** folder and they are ready-to-use.

#### Option 2
All the other training sets could be found at [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).
After extracting the training set by using [rec2image.py](https://github.com/deepinsight/insightface/blob/0b5cab57b6011a587386bb14ac01ff2d74af1ff9/recognition/common/rec2image.py),
using [file_path_extractor.py](https://github.com/HaiyuWu/useful_tools/blob/main/file_path_extractor.py) to gather the image paths of the training set.
Then run [imagelist2lmdb.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/imagelist2lmdb.py) to finish the training set preparation.
```
python3 utils/imagelist2lmdb.py \
--image_list file/of/extracted/image/paths
--destination ../datasets
--file_name dataset/name
```
### Test sets
LFW, CFP-FP, CALFW, CPLFW, AgeDB-30 can be downloaded [here](https://drive.google.com/file/d/1l7XmqzIZKdKVqu0cOS2EI0bL_9_-wIrc/view?usp=drive_link).
Then you can simply run [prepare_test_images.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/prepare_test_images.py) to get datasets ready to test
```
python3 utils/prepare_test_images.py \
--xz_folder folder/contains/xz/files
--destination ../test_set_package_5
--datasets lfw cfp_fp agedb_30 calfw cplfw
```

## Train your own model
After finishing the training and testing sets preparation, you train your own model by:
```
torchrun --nproc_per_node=4 train.py --config_file ./configs/arcface_r100.py
```
You can change the settings at [configs](https://github.com/HaiyuWu/SOTA-FR-train-and-test/tree/main/configs).

## Test your own model
```
python3 test.py \
--model_path path/of/the/weights \
--model iresnet \
--depth 100 \
--val_list lfw cfp_fp agedb_30 calfw cplfw \
--val_source ./test_sets
```

## Feature extraction
Using [file_path_extractor.py](https://github.com/HaiyuWu/useful_tools/blob/main/file_path_extractor.py) to collect the paths of the target images, then run following command to extract the features.
```
python3 feature_extractor.py \
--model_path path/of/the/weights \
--model iresnet \
--depth 100 \
--image_paths image/path/file \
--destination feature/destination
```

## Training results
|                      | Dataset |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |
|----------------------|:-------:|:-----:|:------:|:-----:|:--------:|:-----:|
| CosFace-R100 (m=0.35) | MS1MV2  | 99.83 | 98.43  | 92.92 |  98.28   | 96.07 |
| ArcFace-R100 (m=0.5) | MS1MV2  | 99.78 | 98.20  | 93.15 |  98.13   | 96.03 |
| AdaFace-R100 (m=0.4) | MS1MV2  | 99.82 | 98.34  | 93.05 |  98.17   | 96.10 |

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

## TODO list
Functions:
- [ ] conda env installation
- [ ] resume from training
- [x] train with vit
- [x] feature extraction script
- [x] partial FC
- [x] distributed training

Methods:
- [x] CosFace
- [x] SphereFace
- [x] ArcFace
- [x] ArcFace - combined margin
- [x] AdaFace
- [ ] Circle loss
- [x] MagFace

Backbones:
- [x] iresnet (18, 34, 50, 100, 152, 200)
- [ ] irse
- [x] vit

Others:
- [ ] test on IJB-B, IJB-C, XQLFW, SLLFW, MLFW, TALFW, DoppelVer
- [ ] Published papers
- [ ] references

## License
[MIT license](./license.md)
