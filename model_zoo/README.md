# Model Zoo
Models are trained with ResNet100 backbone and three (**MS1MV2, WebFace4M, and Glint360K**) training sets.

**All the models here are trained with RBG images and ```mean=0.5, std=0.5``` image normalization, which is different from the setting of the original AdaFace and MagFace.
Also, the accuracies are calculated with the merged features of the original and horizontally flipped images, where the original MagFace does not do it. These could cause the difference of the paper-reported accuracy and the accuracy in these tables.** 

| MS1MV2                      |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| CosFace-R100 (m=0.35)       |  MS1MV2   | 99.75 | 98.41  | 93.25 |  98.33   | 95.93 | [Gdrive](https://drive.google.com/file/d/1FzsD117ESm7RDXE2DMvoQmAdkL7W2sa0/view?usp=drive_link) |
| SphereFace-R100 (m=1.7)     |  MS1MV2   | 99.83 | 98.40  | 92.67 |  98.23   | 96.07 | [Gdrive](https://drive.google.com/file/d/1vCMSDF65bslXcU0kxCadBrdW48v-7oPt/view?usp=drive_link) |
| ArcFace-R100 (m=0.5)        |  MS1MV2   | 99.80 | 98.49  | 93.35 |  98.00   | 96.05 | [Gdrive](https://drive.google.com/file/d/1MH2eCU_II2nUtkDHgyD0Scxua1MDZtdE/view?usp=drive_link) |
| ArcFace-R100 (combined)     |  MS1MV2   | 99.80 | 98.50  | 93.17 |  98.07   | 96.18 | [Gdrive](https://drive.google.com/file/d/13K1loXa_YWhkXRSaNXuEqbrCT74nLsoB/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) |  MS1MV2   | 99.78 | 98.44  | 92.95 |  98.05   | 96.08 | [Gdrive](https://drive.google.com/file/d/1OPTjbvgBnVBVrttJKvOzevG9sNQXjA1j/view?usp=drive_link) |
| CircleLoss-R100             |  MS1MV2   |   -   |   -    |   -   |    -     |   -   |                                                -                                                |
| MagFace-R100                |  MS1MV2   | 99.82 | 98.21  | 92.67 |  98.15   | 96.13 | [Gdrive](https://drive.google.com/file/d/1h_V93Sc1NB5eLW26-pB7KCB7-BSCYHZj/view?usp=drive_link) |
| AdaFace-R100 (m=0.4)        |  MS1MV2   | 99.82 | 98.63  | 93.05 |  98.20   | 96.15 | [Gdrive](https://drive.google.com/file/d/1a0BkAUwFC8O_sR2cW0NOM93zgOKBWbsr/view?usp=drive_link) |
| UniFace-R100 (m=0.5)        |  MS1MV2   | 99.78 | 98.49  | 93.28 |  98.02   | 96.10 | [Gdrive](https://drive.google.com/file/d/1TgO7RgXPoMoM6ESIj7h09WYhotou47vD/view?usp=drive_link) |


|          WebFace4M          |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| ArcFace-R100                | WebFace4M | 99.77 | 99.21  | 94.27 |  97.85   | 96.12 | TODO |
| ArcFace-R100 (combined)     | WebFace4M | 99.77 | 99.00  | 94.42 |  97.90   | 96.13 | [Gdrive](https://drive.google.com/file/d/1DXoYmNi5O2U_HF6vj4WKNHPEpWYNaw5O/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) | WebFace4M | 99.82 | 99.04  | 94.30 |  97.93   | 95.98 | TODO |
| MagFace-R100                | WebFace4M | 99.80 | 99.23  | 94.18 |  97.88   | 95.97 | TODO |
| AdaFace-R100 (m=0.4)        | WebFace4M | 99.78 | 99.14  | 94.32 |  97.63   | 96.13 | [Gdrive](https://drive.google.com/file/d/1YRqrXGOao5F3mVQXZ90dt-WLEICNqbYY/view?usp=drive_link) |
| UniFace-R100 (m=0.5)        | WebFace4M | 99.77 | 99.21  | 94.47 |  97.60   | 96.02 | TODO |

|          Glint360k          |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| ArcFace-R100 (m=0.5)        | Glint360k | 99.82 | 99.07  | 94.68 |  98.30   | 96.15 | TODO |
| CurricularFace-R100 (m=0.5) | Glint360k | 99.78 | 99.11  | 94.68 |  98.43   | 96.17 | TODO |
| MagFace-R100                | Glint360k | 99.82 | 99.14  | 94.47 |  98.20   | 96.15 | TODO |
| AdaFace-R100 (m=0.4)        | Glint360k | 99.78 | 99.22  | 95.05 |  98.35   | 96.07 | TODO |
| UniFace-R100 (m=0.5)        | Glint360k | 99.73 | 99.09  | 94.58 |  98.22   | 96.18 | TODO |
