# 📒VIP
- 캡스톤디자인 - 최우수상 <br/>
- ViTGAN, PCVOS, OpenCV, Pyqt5를 활용한 Video Inpainting 프로그램 제작
  
### 📒시연 영상

https://github.com/hyul77/VIP/assets/100561170/94f8753f-c21e-4896-bd42-d9fc225d8e32


## 📒설명
- VideoInpainting을 활용하기 위해 Video Object Segmentation 모델인 PCVOS를 사용
- PCVOS를 실행시키기 위해 첫 Frame masking 하는 기능을 OpenCV를 통해 구현


## 📒 개발 목적 및 목표
 본 과제는 기억하고 싶은 순간을 동영상으로 촬영했는데 원치 않는 사물이나 사람이 같이 찍히는 경우 그러한 물체 즉, 동영상 속 원하는 객체를 지정해 Masking을 해 손상시켜준다. 그리고 GAN을 통해 손상된 이미지를 메꾼 생성된 각 frame별 이미지를 제공해 동영상에 대한 만족도, 초상권 문제 회피, 동영상 결함 복구 등의 문제를 해결하고자 한다.
<br/><br/>
 이 기술을 통해 Image에만 적용되던 AI 지우개와 같은 편집 기술과 초상권을 회피하기 위해 video 위에 모자이크 처리를 하는 등의 편집 기술을 대신해 해당 동영상에서 원하는 물체를 Masking을 통해 손상시킨 후 채워 넣는 형식인 Inpainting을 돌입해 원본에 손상을 입히는 편집 기술이 아닌 생성하는 형태가 적용된다.
 
<br/>



## 📒 주요 기능
![image](https://github.com/hyul77/VIP/assets/100561170/b31d41ea-a61e-40bd-89d8-bbdfe88bb599)
![image](https://github.com/hyul77/VIP/assets/100561170/b04d03d3-b3a0-481f-8edb-d1302068b0c9)

<br/>

## 📒 기술 스택
- [ViTGAN](https://github.com/wilile26811249/ViTGAN)<br/>
- [VideoInpainting](https://github.com/ruiliu-ai/FuseFormer)<br/>
- [PCVOS](https://github.com/pkyong95/PCVOS)<br/>

<br/>

## 📒 기대효과 및 활용분야
 현재 갤럭시 휴대폰의 갤러리에는 AI 지우개를 제공하고 있다. 현재로선 이미지에만 적용이 되고 있는데 이 프로그램은 이미지가 아닌 Video로 제공된다. 새로운 영상 편집 기술로 인해 여태 활용하지 못했던 부분들에 활용을 할 수 있게 된다.
<br/><br/>
예시를 들자면 
1) 원치 않는 사물이나 사람이 같이 찍히는 경우 그러한 물체 즉, 동영상 속 원하는 객체를 제거할 수 있게 된다.<br/>
2) 동영상 원본에 손상을 입히는 기술인 모자이크와 같은 것들을 대체할 수 있다.<br/>
3) 의료 관련 동영상과 같은 동영상이 손상된 경우 그 부분을 복원할 수 있다.<br/>
<br/><br/>

기대효과로는
1) 모자이크와 같은 기능을 이 기술로 대체 가능하다.<br/>
2) 동영상을 원하는 대로 수정해 만족도를 높일 수 있다.<br/>
3) 초상권과 같은 문제를 해결할 수 있다.<br/>
4) 초상권에 민감한 영상을 다루는 사람들이 이 기술을 쓸 것을 예측할 수 있다.<br/>
<br/>






