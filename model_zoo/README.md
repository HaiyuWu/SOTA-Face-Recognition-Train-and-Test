# Model Zoo
Models are trained with ResNet100 backbone and three (**MS1MV2, WebFace4M, and Glint360K**) training sets.

|            MS1MV2           |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| CosFace-R100 (m=0.35)       |  MS1MV2   | 99.83 | 98.44  | 92.92 |  98.28   | 96.07 | [Gdrive](https://drive.google.com/file/d/1FzsD117ESm7RDXE2DMvoQmAdkL7W2sa0/view?usp=drive_link) |
| SphereFace-R100 (m=1.7)     |  MS1MV2   | 99.83 | 98.20  | 92.77 |  98.08   | 96.13 | [Gdrive](https://drive.google.com/file/d/1vCMSDF65bslXcU0kxCadBrdW48v-7oPt/view?usp=drive_link) |
| ArcFace-R100 (m=0.5)        |  MS1MV2   | 99.78 | 98.20  | 93.15 |  98.13   | 96.03 | [Gdrive](https://drive.google.com/file/d/1MH2eCU_II2nUtkDHgyD0Scxua1MDZtdE/view?usp=drive_link) |
| ArcFace-R100 (combined)     |  MS1MV2   | 99.82 | 98.07  | 92.60 |  97.95   | 96.18 | [Gdrive](https://drive.google.com/file/d/13K1loXa_YWhkXRSaNXuEqbrCT74nLsoB/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) |  MS1MV2   | 99.80 | 98.26  | 92.50 |  97.97   | 96.03 | [Gdrive](https://drive.google.com/file/d/1OPTjbvgBnVBVrttJKvOzevG9sNQXjA1j/view?usp=drive_link) |
| CircleLoss-R100             |  MS1MV2   |   -   |   -    |   -   |    -     |   -   |                                                -                                                |
| MagFace-R100                |  MS1MV2   | 99.83 | 98.20  | 92.33 |  98.07   | 96.10 | [Gdrive](https://drive.google.com/file/d/1h_V93Sc1NB5eLW26-pB7KCB7-BSCYHZj/view?usp=drive_link) |
| AdaFace-R100 (m=0.4)        |  MS1MV2   | 99.82 | 98.34  | 93.05 |  98.17   | 96.10 | [Gdrive](https://drive.google.com/file/d/1a0BkAUwFC8O_sR2cW0NOM93zgOKBWbsr/view?usp=drive_link) |
| UniFace-R100 (m=0.5)         |  MS1MV2   | 99.78 | 98.31  | 93.03 |  98.03   | 96.15 | [Gdrive](https://drive.google.com/file/d/1TgO7RgXPoMoM6ESIj7h09WYhotou47vD/view?usp=drive_link) |


|          WebFace4M          |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| MagFace-R50                 | WebFace4M | 99.73 | 98.77  | 93.48 |  97.72   | 95.97 | [Gdrive](https://drive.google.com/file/d/1ExTFjubgP5rRVhIkOa7aRBRgwSvKJQoL/view?usp=drive_link) |
| ArcFace-R100                | WebFace4M | 99.80 | 99.11  | 94.37 |  97.82   | 96.08 | TODO |
| ArcFace-R100 (combined)     | WebFace4M | 99.77 | 99.06  | 94.05 |  97.83   | 96.03 | [Gdrive](https://drive.google.com/file/d/1DXoYmNi5O2U_HF6vj4WKNHPEpWYNaw5O/view?usp=drive_link) |
| CurricularFace-R100 (m=0.5) | WebFace4M | 99.80 | 99.03  | 94.08 |  97.82   | 96.02 | TODO |
| MagFace-R100                | WebFace4M | 99.82 | 99.11  | 94.20 |  97.83   | 96.03 | TODO |
| AdaFace-R100 (m=0.4)        | WebFace4M | 99.80 | 99.03  | 94.48 |  97.72   | 96.08 | [Gdrive](https://drive.google.com/file/d/1YRqrXGOao5F3mVQXZ90dt-WLEICNqbYY/view?usp=drive_link) |
| UniFace-R100 (m=0.5)        | WebFace4M | 99.78 | 99.16  | 94.20 |  97.70   | 96.10 | TODO |

|          Glint360k          |  Dataset  |  LFW  | CFP-FP | CPLFW | AgeDB-30 | CALFW |                                             Weights                                             |
|-----------------------------|:---------:|:-----:|:------:|:-----:|:--------:|:-----:|:-----------------------------------------------------------------------------------------------:|
| ArcFace-R100 (m=0.5)        | Glint360k | 99.82 | 99.09  | 94.67 |  98.45   | 96.13 | TODO |
| CurricularFace-R100 (m=0.5) | Glint360k | 99.82 | 99.21  | 94.53 |  98.50   | 96.05 | TODO |
| MagFace-R100                | Glint360k | 99.80 | 99.00  | 94.63 |  98.37   | 96.23 | TODO |
| AdaFace-R100 (m=0.4)        | Glint360k | 99.80 | 99.24  | 94.92 |  98.43   | 96.03 | TODO |
| UniFace-R100 (m=0.5)        | Glint360k | 99.73 | 98.90  | 94.65 |  98.25   | 96.12 | TODO |
