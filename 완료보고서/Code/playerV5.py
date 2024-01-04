import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtGui import *
import subprocess

os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "directshow"

class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super(VideoPlayer, self).__init__(parent)
        self.video_path = None
        self.setWindowTitle("Video Player")
        self.setGeometry(100,100,800,600)
        #initialize file path
        self.selectedFilePath = None

        self.setStyleSheet('background-color: #848484;')

        #set video player
        self.videoWidget = QVideoWidget()
        self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        
        #initialize volume slider
        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.setRange(0,100)
        self.volumeSlider.setValue(50)
        self.volumeSlider.valueChanged.connect(self.setVolume)

        
        self.mediaPlayer.durationChanged.connect(self.updateDuration)
        self.mediaPlayer.positionChanged.connect(self.updatePosition)

        self.playbackPosition = 0

        self.durationLabel = QLabel('00:00:00')
        self.positionLabel = QLabel('00:00:00  / ')
        
        
        selectIcon = QIcon(QPixmap('/home/hyul/Desktop/ViVE/Icon/selectImage.png').scaled(22,22))

        selectButton = QPushButton()
        selectButton.setIcon(selectIcon)
        selectButton.clicked.connect(self.openFile)
        selectButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')
        
        
        self.playIcon = QIcon(QPixmap('/home/hyul/Desktop/ViVE/Icon/playImage.png').scaled(22,22))
        self.pauseIcon = QIcon(QPixmap('/home/hyul/Desktop/ViVE/Icon/pauseImage.png').scaled(22,22))

        self.playButton = QPushButton()
        self.playButton.setIcon(self.playIcon)
        self.playButton.clicked.connect(self.playVideo)
        self.playButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')

        stopButton = QPushButton("Stop")
        stopButton.clicked.connect(self.stopVideo)
        stopButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')

        self.pauseButton = QPushButton()
        self.pauseButton.setIcon(self.pauseIcon)
        self.pauseButton.clicked.connect(self.pauseVideo)
        self.pauseButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')


        cutButton = QPushButton("Cut")
        #cutButton.clicked.connect(self.)
        cutButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')

        captionButton = QPushButton("Caption")
        #captionButton.clicked.connect(self.)
        captionButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: #ffffff; border-radius: 10px; padding: 10px; }')


        
        font = QFont()
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1)
        font.setPixelSize(20)
        font.setBold(True)
        
        maskingButton = QPushButton("Masking")
        maskingButton.setFont(font)
        maskingButton.clicked.connect(self.draw)
        maskingButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')
        
        trackingButton = QPushButton("Tracking")
        trackingButton.setFont(font)
        trackingButton.clicked.connect(self.masking)
        trackingButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')

        inpaintingButton = QPushButton("Inpainting")
        inpaintingButton.setFont(font)
        inpaintingButton.clicked.connect(self.inpainting)
        inpaintingButton.setStyleSheet('QPushButton { background-color: #ADADAD; color: black; border-radius: 10px; padding: 10px; }')

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.playButton)
        buttonLayout.addWidget(self.pauseButton)
        buttonLayout.addWidget(selectButton)
        
        volumeLabel = QLabel()
        volumeLabel.setPixmap(QPixmap('/home/hyul/Desktop/ViVE/Icon/volumeImage.png').scaled(22,22))

        volumeLayout = QHBoxLayout()
        volumeLayout.addWidget(volumeLabel)
        volumeLayout.addWidget(self.volumeSlider)

        labelSliderLayout = QHBoxLayout()
        labelSliderLayout.addWidget(self.positionLabel)
        labelSliderLayout.addWidget(self.durationLabel)
        labelSliderLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addLayout(buttonLayout)
        layout.addLayout(labelSliderLayout)
        layout.addLayout(volumeLayout)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(maskingButton)
        leftLayout.addWidget(trackingButton)
        leftLayout.addWidget(inpaintingButton)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout)
        mainLayout.addLayout(layout)

        self.setLayout(mainLayout)




    def playVideo(self):
        if self.selectedFilePath:
            playIcon2 = QIcon(QPixmap('/home/hyul/Desktop/ViVE/Icon/playImage2.png').scaled(22,22))
            self.playButton.setIcon(playIcon2)
            self.pauseButton.setIcon(self.pauseIcon)
            self.mediaPlayer.play()
        else:
            self.showMessageBox("File not selected", "Please select a video file before playing.")


    def stopVideo(self):
        self.playbackPosition = 0
        self.mediaPlayer.stop()


    def pauseVideo(self):
        pauseIcon2 = QIcon(QPixmap('/home/hyul/Desktop/ViVE/Icon/pauseImage2.png').scaled(22,22))
        self.pauseButton.setIcon(pauseIcon2)
        self.playButton.setIcon(self.playIcon)
        self.playbackPosition = self.mediaPlayer.position()
        self.mediaPlayer.pause()


    def setVolume(self, volume):
        self.mediaPlayer.setVolume(volume)


    def updateDuration(self, duration):
        self.positionSlider.setMaximum(duration)


    def updatePosition(self, position):
        self.positionSlider.setValue(position)

        duration_time = self.mediaPlayer.duration() / 1000
        duration_minutes = duration_time // 60
        duration_seconds = duration_time % 60
        duration_milliseconds = (duration_time % 1) * 100
        self.durationLabel.setText(f'{int(duration_minutes):02d}:{int(duration_seconds):02d}:{int(duration_milliseconds):02d}')

        position_time = position / 1000
        position_minutes = position_time // 60
        position_seconds = position_time % 60
        position_milliseconds = position % 100
        self.positionLabel.setText(f'{int(position_minutes):02d}:{int(position_seconds):02d}:{int(position_milliseconds):02d}  / ')


    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    
    def openFile(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *mkv)", options=options)

        if fileName:
            self.selectedFilePath = fileName
            self.video_path = fileName
            media = QMediaContent(QUrl.fromLocalFile(fileName))
            self.mediaPlayer.setMedia(media)
        else:
            print("FIle selection canceled or invalid file.")
    

    def draw(self):
        print("draw")
        command = "python"
        parameters = ["./masking/draw7.py", self.selectedFilePath]

     # QProcess 인스턴스 생성
        self.process = QProcess(self)

    # 표준 출력 및 표준 에러 채널을 설정
        self.process.setReadChannel(QProcess.StandardOutput)

    # 외부 프로세스의 출력 및 에러를 읽는 함수를 연결
        self.process.readyReadStandardOutput.connect(self.handleDrawOutput)
        self.process.readyReadStandardError.connect(self.handleDrawError)

    # 실행 중이 아니면 시작
        self.process.start(command, parameters)

    # QProcess의 상태 변화를 감시하고 완료되면 처리
        self.process.finished.connect(lambda exitCode, exitStatus: self.processFinished(self.process, exitCode, exitStatus))
        
    def handleDrawOutput(self):
            # 외부 프로세스의 표준 출력을 읽어와서 출력
            output = self.process.readAllStandardOutput().data().decode("utf-8")
            print(output)
            
    def handleDrawError(self):
    # 외부 프로세스의 표준 에러 출력을 읽어와서 출력
            error_output = self.process.readAllStandardError().data().decode("utf-8")
            print("Error output:", error_output)

    def processFinished(self, process, exitCode, exitStatus):
	    # 외부 프로세스가 종료되면 호출되는 함수
    	print("Process finished with exit code:", exitCode)
    	print("Exit status:", exitStatus)
    
    	
    def masking(self):
    	print("masking")
    	command = "CUDA_VISIBLE_DEVICES=0 python ./masking/masking.py --clip_length 5 --refine_clip ICR --T_window 2 --S_window 7 --shared_proj --memory_read PMM --predict_all --time"
    	process = QProcess(self)
    	process.start(command)
    	process.waitForFinished()
    	subprocess.run(command, shell=True)
    	
    def inpainting(self):
    	print("check")
    	video_name = os.path.splitext(os.path.basename(self.video_path))[0]
    	command = f'python ./inpainting/inpainting.py -c ./inpainting/checkpoints/ViV_davis/gen_00053.pth -v ./inputdata/JPEGImages/{video_name} -m ./inputdata/Annotation/{video_name}'
    	process = QProcess(self)
    	process.start(command)
    	process.waitForFinished()
    	subprocess.run(command, shell=True)
    	


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
