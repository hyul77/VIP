import cv2
import os

def images_to_video(image_folder, video_name, fps=15):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x[:-4]))  # 파일 이름의 숫자 순으로 정렬
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# 사용 예제
images_folder = "/home/hyul/Desktop/ViVE/inpainting/data/DAVIS/JPEGImages/tennis"
video_output = "/home/hyul/Desktop/ViVE/tennis.mp4"
images_to_video(images_folder, video_output)


# data 파일들은 삭제하면 안돼 zz 헐 저 뭐 잘못 삭제했어요?

# 아까 bike 폴더 삭제한거 아니야? 아 그거 파일 베어 폴더 복사한거 ㅇㅎ바이크로 이름 바꾸고 베어 프레임들 삭제했어요 ㅋㅋㅋ
# 순서 헷갈리면 말해 다시 설명 가능하니까ㅡ 형 지금 전화 가능해요/?W?K??잠만 폰 가져올께
