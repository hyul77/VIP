import os
import sys
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import subprocess

os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "directshow"

class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.frame = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.pixmap = None
        self.mask_frame = None  # 추가된 부분: 검은 화면
        self.file_path = sys.argv[1]
        self.initUI()
        
        self.setStyleSheet('background-color: #848484;')

    def initUI(self):
        self.setWindowTitle('Video Editor')
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(1000,700)
        
        font = QFont()
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1)
        font.setPixelSize(20)
        font.setBold(True)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.capture_button = QPushButton('Capture Frame', self)
        self.capture_button.setFont(font)
        self.capture_button.clicked.connect(self.captureFrame)
        self.capture_button.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')

        self.mask_button = QPushButton('Masking', self)
        self.mask_button.setFont(font)
        self.mask_button.clicked.connect(self.maskFrame)
        self.mask_button.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')
    
        self.finish_button = QPushButton('Finish', self)
        self.finish_button.setFont(font)
        self.finish_button.clicked.connect(self.finish)
        self.finish_button.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.mask_button)
        layout.addWidget(self.finish_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        
    def openVideo(self):

        if self.file_path:
            self.video_path = self.file_path
            self.video_name = os.path.basename(self.file_path)
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.updateFrame()
            
            # 환경 변수 설정
            os.environ['VIDEO_PATH'] = self.video_path
            os.environ['VIDEO_NAME'] = self.video_name
            
            command = "python ./masking/all_frame_cut.py"
            process = QProcess(self)
            process.start(command)
            process.waitForFinished()
            subprocess.run(command, shell=True)


    def updateFrame(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate the scaled frame to fit in the window
                label_size = self.video_label.size()
                frame_ratio = frame.shape[1] / frame.shape[0]
                label_ratio = label_size.width() / label_size.height()

                if frame_ratio > label_ratio:
                    scaled_width = label_size.width()
                    scaled_height = int(label_size.width() / frame_ratio)
                else:
                    scaled_width = int(label_size.height() * frame_ratio)
                    scaled_height = label_size.height()

                # Resize the frame
                frame = cv2.resize(frame, (scaled_width, scaled_height))

                bytes_per_line = 3 * scaled_width
                q_image = QImage(frame.data, scaled_width, scaled_height, bytes_per_line, QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(q_image)
                self.video_label.setPixmap(self.pixmap)

                # 추가된 부분: 검은 화면 초기화
                self.mask_frame = np.zeros_like(frame)

    def captureFrame(self):
        print("capture")
        sys.stdout.flush()   
        if self.mask_frame is not None:
            video_name_without_extension = os.path.splitext(os.path.basename(self.video_path))[0]
            # Save the QPixmap (with user's drawings) on the black frame
            os.makedirs(f'inputdata/Annotations/{video_name_without_extension}', exist_ok=True)
            #self.pixmap.save(f'inputdata/Annotations/{self.video_name}/{self.current_frame:05d}.png', 'PNG')
            
            cv2.imwrite(f'inputdata/Annotations/{video_name_without_extension}/00000.png', self.mask_frame)

    def maskFrame(self):
        self.openVideo()
        if self.frame is not None:
            self.drawing = True
            self.last_point = None

    def mousePressEvent(self, event):
        if self.drawing:
            video_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_ratio = video_size[0] / video_size[1]
            label_size = self.video_label.size()
            label_ratio = label_size.width() / label_size.height()

            if video_ratio > label_ratio:
                scaled_width = label_size.width()
                scaled_height = int(label_size.width() / video_ratio)
            else:
                scaled_width = int(label_size.height() * video_ratio)
                scaled_height = label_size.height()

            self.scaled_offset_x = (label_size.width() - scaled_width) // 2
            self.scaled_offset_y = (label_size.height() - scaled_height) // 2

            self.last_point = event.pos() - self.video_label.pos() - QPoint(self.scaled_offset_x, self.scaled_offset_y)
            self.pixmap = self.video_label.pixmap()  # 현재 QPixmap 가져오기
            self.painter = QPainter(self.pixmap)  # QPainter 객체 생성
            self.painter.setPen(QPen(QColor(255, 0, 0, 255), 10, Qt.SolidLine, Qt.RoundCap))  # Alpha 값을 추가하여 오류 방지
            self.painter.begin(self.pixmap)  # 그림 그리기 시작

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.painter.end()  # 그림 그리기 종료

    def mouseMoveEvent(self, event):
        try:
            if self.drawing and self.last_point:
                current_point = event.pos() - self.video_label.pos() - QPoint(self.scaled_offset_x,
                                                                              self.scaled_offset_y)
                self.painter.drawLine(self.last_point, current_point)
                self.video_label.setPixmap(self.pixmap)  # Set the painted QPixmap to the QLabel

                # 추가된 부분: Paint the drawn part on the black frame as well
                cv2.line(self.mask_frame, tuple((self.last_point.x(), self.last_point.y())),
                         tuple((current_point.x(), current_point.y())), (255, 255, 255), 10)

                self.last_point = current_point
        except Exception as e:
            print(f"Error in mouseMoveEvent: {e}")
    
    def finish(self):
        self.close()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = VideoEditor()
    editor.show()
    sys.exit(app.exec_())



