# all_frame_cut.py

import cv2
import os

# 환경 변수에서 정보 가져오기
file_path = os.environ.get('VIDEO_PATH')
video_name = os.environ.get('VIDEO_NAME')
# 파일 이름에서 확장자 제외
video_name_without_extension, _ = os.path.splitext(video_name)

# 동영상 파일 열기
cap = cv2.VideoCapture(file_path)

# 프레임 수 얻기
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 폴더 생성
os.makedirs(f'inputdata/JPEGImages/{video_name_without_extension}', exist_ok=True)

# 프레임 단위로 이미지 추출
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # 이미지 파일로 저장
    frame_filename = os.path.join(f'inputdata/JPEGImages/{video_name_without_extension}', f'{i:05d}.jpg')
    cv2.imwrite(frame_filename, frame)

# 동영상 파일 닫기
cap.release()

