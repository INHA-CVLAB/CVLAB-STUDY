### CVLAB-STUDY  
  
  Networks Analysis  
- [x] SqueezeNet model structure explain and code review -18.12.21-  
- [x] MobileNetV1,2 model structure explain and code review   
- [x] IncpetionV1,2 model structure explain and code review  
    
 
 Detection Alogrithm Analysis_Tensorflow  
- [x] SSD.Tensorflow Loss function explain and code review   
- [x] Decoder part explain 
- [x] Encoder part explain 
  
 Train  
- [x] Train by Pascal VOC(2007, 2012) and test(2007) : mAP 77.8% ([For detail](https://github.com/INHA-CVLAB/CVLAB-STUDY/wiki)) -19.01.25-
- [ ] Train by Kitti and test
- [ ] Train by msCOCO and test  

   
 Detection Alogrithm Analysis_Keras  
- [ ] SSD.Tensorflow Loss function explain and code review   
- [ ] Decoder part explain 
- [ ] Encoder part explain 

  
    
---
|VOC0712 / msCOCO |SSD + MobileNet|SSD + SqueezeNet| SSD + Inception |  SSD + VGG(Original)| Pelee(SOTA) |
|----|:----:|:----:|:----:|:----:|:----:|
|# parameters| 6.8M | - | - | 34.3M | 5.4M |
|Expected|mAP=0.727  |mAP=0.643  |mAP=0.782 |mAP=0.778 |mAP=76.4 / 22.4 |
|Result|  |  |  |  | |
|FPS(Intel i7)| | | | | |

|VOC0712 |Tiny-YOLOv2|YOLOv2| 
|----|:----:|:----:|
|# parameters| 15.9M | 58M | 
|Expected|mAP=0.571  |mAP=0.69  |
|Result| - | - |
|FPS(Intel i7)| - | - |
