import os
import sys
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
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Editor')
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(800,500)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.capture_button = QPushButton('Capture Frame', self)
        self.capture_button.clicked.connect(self.captureFrame)

        self.mask_button = QPushButton('Masking', self)
        self.mask_button.clicked.connect(self.maskFrame)

        self.cut_button = QPushButton('Cut', self)
        self.cut_button.clicked.connect(self.cutVideo)

        open_action = QAction('Open Video', self)
        open_action.triggered.connect(self.openVideo)
        open_action.setShortcut('Ctrl+O')

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(open_action)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.mask_button)
        layout.addWidget(self.cut_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def openVideo(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.updateFrame()

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

    def captureFrame(self):
        if self.frame is not None:
            # Save the QPixmap (with user's drawings)
            self.pixmap.save(f'captured_frame_{self.current_frame}.png', 'PNG')

    def maskFrame(self):
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
            self.painter.setPen(QPen(QColor(0, 0, 255), 10, Qt.SolidLine, Qt.RoundCap))
            self.painter.begin(self.pixmap)  # 그림 그리기 시작

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point:
            current_point = event.pos() - self.video_label.pos() - QPoint(self.scaled_offset_x, self.scaled_offset_y)
            self.painter.drawLine(self.last_point, current_point)
            self.video_label.setPixmap(self.pixmap)  # 그린 QPixmap을 QLabel에 설정
            self.last_point = current_point

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.painter.end()  # 그림 그리기 종료

    def cutVideo(self):
        if self.video_path is not None and self.cap is not None:
            start_frame = 10  # 자르기 시작할 프레임 번호
            end_frame = 50  # 자르기를 종료할 프레임 번호
            output_path = 'output.mp4'  # 출력될 동영상 파일의 경로

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (int(self.cap.get(3)), int(self.cap.get(4))))

            for frame_num in range(start_frame, end_frame + 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                if ret:
                    out.write(frame)

            out.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = VideoEditor()
    editor.show()
    sys.exit(app.exec())

