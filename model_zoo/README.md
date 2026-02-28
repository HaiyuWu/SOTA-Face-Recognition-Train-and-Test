# Model Zoo
Models are trained with ResNet100 backbone and three (**MS1MV2, WebFace4M, and Glint360K**) training sets.

**All the models here are trained with RBG images and ```mean=0.5, std=0.5``` image normalization, which is different from the setting of the original AdaFace and MagFace.
Also, the accuracies are calculated with the merged features of the original and horizontally flipped images, where the original MagFace does not do it. These could cause the difference of the paper-reported accuracy and the accuracy in these tables.**

| Model                         |  Dataset  | LFW | CFP-FP | CPLFW | AgeDB-30 | CALFW | Hadrian | Eclipse | Twins | Weights |
|:------------------------------|:---------:|:------:|:------:|:-----:|:--------:|:-----:|:-------:|:-------:|:------:|:------------------------------------------------------------------------------------------------:
| AdaFace-R100 (m=0.4)       | MS1MV2    | 99.82 | 98.63 | 93.05 | 98.13 | 96.15 | 92.35 | 82.32 | 68.57 | [Gdrive](https://drive.google.com/file/d/1a0BkAUwFC8O_sR2cW0NOM93zgOKBWbsr/view?usp=drive_link) |
| ArcFace-R100 (m=0.5)       | MS1MV2    | 99.80 | 98.49 | 93.35 | 98.00 | 96.05 | 91.72 | 81.80 | 68.28 | [Gdrive](https://drive.google.com/file/d/1MH2eCU_II2nUtkDHgyD0Scxua1MDZtdE/view?usp=drive_link) |
| ArcFace-R100 (combined)    | MS1MV2    | 99.80 | 98.53 | 93.15 | 98.07 | 96.18 | 92.88 | 83.13 | 68.40 | [Gdrive](https://drive.google.com/file/d/1Uznh1O0EJoD34A3YchvI7FQ1G67rNGKP/view?usp=drive_link) |
| CosFace-R100 (m=0.35)      | MS1MV2    | 99.75 | 98.41 | 93.25 | 98.37 | 95.93 | 92.33 | 82.85 | 67.38 | [Gdrive](https://drive.google.com/file/d/1FzsD117ESm7RDXE2DMvoQmAdkL7W2sa0/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) | MS1MV2    | 99.78 | 98.29 | 92.88 | 98.05 | 96.08 | 91.48 | 82.10 | 66.00 | [Gdrive](https://drive.google.com/file/d/1OPTjbvgBnVBVrttJKvOzevG9sNQXjA1j/view?usp=drive_link) |
| MagFace-R100               | MS1MV2    | 99.82 | 98.40 | 92.95 | 98.22 | 96.02 | 92.82 | 82.67 | 68.55 | [Gdrive](https://drive.google.com/file/d/1h_V93Sc1NB5eLW26-pB7KCB7-BSCYHZj/view?usp=drive_link) |
| SphereFace-R100 (m=1.7)    | MS1MV2    | 99.83 | 98.40 | 92.88 | 98.20 | 96.07 | 93.27 | 83.48 | 67.67 | [Gdrive](https://drive.google.com/file/d/1vCMSDF65bslXcU0kxCadBrdW48v-7oPt/view?usp=drive_link) |
| UniFace-R100 (m=0.5)       | MS1MV2    | 99.78 | 98.49 | 93.28 | 98.02 | 96.10 | 91.63 | 82.18 | 65.75 | [Gdrive](https://drive.google.com/file/d/1TgO7RgXPoMoM6ESIj7h09WYhotou47vD/view?usp=drive_link) |

| Model                         |  Dataset  | LFW | CFP-FP | CPLFW | AgeDB-30 | CALFW | Hadrian | Eclipse | Twins | Weights |
|:------------------------------|:---------:|:------:|:------:|:-----:|:--------:|:-----:|:-------:|:-------:|:------:|:-----------------------------------------------------------------------------------------------:|
| AdaFace-R100 (m=0.4)       | WebFace4M | 99.78 | 99.14 | 94.32 | 97.63 | 96.13 | 91.07 | 82.80 | 72.47 | [Gdrive](https://drive.google.com/file/d/19uHspLbfkMv0_HfYC3Ege4HQCIsxo8Vr/view?usp=drive_link) |
| ArcFace-R100 (m=0.5)       | WebFace4M | 99.77 | 99.21 | 94.25 | 97.85 | 96.12 | 91.78 | 83.20 | 71.73 | [Gdrive](https://drive.google.com/file/d/1yzm9-VFyVqm9HkQRXDwlLutF0G4tq86x/view?usp=drive_link) |
| ArcFace-R100 (combined)    | WebFace4M | 99.77 | 99.00 | 94.40 | 97.90 | 96.13 | 91.90 | 83.17 | 71.48 | [Gdrive](https://drive.google.com/file/d/15_i01irmE-ruB00qgiU6nOpyPFwKulNz/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) | WebFace4M | 99.82 | 98.97 | 94.30 | 97.93 | 96.02 | 90.40 | 82.03 | 70.43 | [Gdrive](https://drive.google.com/file/d/1WUOpOReeaUBmmqfQGXAvl3WMUMfoY4MZ/view?usp=drive_link) |
| MagFace-R100               | WebFace4M | 99.80 | 99.23 | 94.25 | 97.88 | 95.97 | 90.92 | 82.47 | 71.08 | [Gdrive](https://drive.google.com/file/d/1rVBzy01b_ZWTUJ97ainjS9tWEc2u26Or/view?usp=drive_link) |
| UniFace-R100 (m=0.5)       | WebFace4M | 99.77 | 99.17 | 94.47 | 97.60 | 96.02 | 90.65 | 82.20 | 71.82 | [Gdrive](https://drive.google.com/file/d/1cfJREzrOEqVUAzD5ptH_htIEZiYWdr9H/view?usp=drive_link) |

| Model                         |  Dataset  | LFW | CFP-FP | CPLFW | AgeDB-30 | CALFW | Hadrian | Eclipse | Twins | Weights |
|:------------------------------|:---------:|:------:|:------:|:-----:|:--------:|:-----:|:-------:|:-------:|:------:|:-----------------------------------------------------------------------------------------------:|
| AdaFace-R100 (m=0.4)       | Glint360k | 99.78 | 99.21 | 94.98 | 98.32 | 96.07 | 95.63 | 83.88 | 75.88 | [Gdrive](https://drive.google.com/file/d/1YRqrXGOao5F3mVQXZ90dt-WLEICNqbYY/view?usp=drive_link) |
| ArcFace-R100 (m=0.5)       | Glint360k | 99.82 | 99.07 | 94.68 | 98.30 | 96.15 | 95.68 | 84.12 | 84.67 | [Gdrive](https://drive.google.com/file/d/1JsRePpJtVjzgv0N-JSUpyt7ikzsNEaS7/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) | Glint360k | 99.78 | 99.11 | 94.80 | 98.40 | 96.17 | 94.63 | 83.53 | 74.23 | [Gdrive](https://drive.google.com/file/d/12pNosc10tOGl-OCH65OH3_rhQ6Le1tMS/view?usp=drive_link) |
| MagFace-R100               | Glint360k | 99.82 | 99.14 | 94.47 | 98.20 | 96.15 | 95.22 | 83.83 | 71.98 | [Gdrive](https://drive.google.com/file/d/1xg7CBPhatTE1BwmozGIOhI1bfGwWiyUP/view?usp=drive_link) |
| UniFace-R100 (m=0.5)       | Glint360k | 99.73 | 99.09 | 94.58 | 98.22 | 96.18 | 93.53 | 83.18 | 71.52 | [Gdrive](https://drive.google.com/file/d/1r_LA7F0rmga_ip1MDtoue5lZYBFbtg1-/view?usp=drive_link) |


# References
- [1] Deng, Jiankang, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." CVPR2019.
- [2] Huang, Yuge, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. "Curricularface: adaptive curriculum learning loss for deep face recognition." CVPR2020.
- [3] Meng, Qiang, Shichao Zhao, Zhida Huang, and Feng Zhou. "Magface: A universal representation for face recognition and quality assessment." CVPR2021.
- [4] Kim, Minchul, Anil K. Jain, and Xiaoming Liu. "Adaface: Quality adaptive margin for face recognition." CVPR2022.
- [5] Zhou, Jiancan, Xi Jia, Qiufu Li, Linlin Shen, and Jinming Duan. "UniFace: Unified Cross-Entropy Loss for Deep Face Recognition." ICCV2023.
- [6] Wang, Hao, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. "Cosface: Large margin cosine loss for deep face recognition." CVPR2018.
- [7] Liu, Weiyang, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. "Sphereface: Deep hypersphere embedding for face recognition." CVPR2017.
