import time

import cv2
from .throttle import Throttle

class Camera:
    def __init__(self, ip: str, port: str, username: str, password: str, channel: str):
        self.ip = ip
        self.port = port
        self.__username = username
        self.__password = password
        self.channel = channel
        self.__capture = cv2.VideoCapture(self.__rtsp_url, cv2.CAP_FFMPEG)
        self.__hog = cv2.HOGDescriptor()
        self.has_stopped = False
        self.__hog.setSVMDetector(self.__hog.getDefaultPeopleDetector())
        self.throttle = Throttle(interval_secs=0.1)

    @property
    def __rtsp_url(self):
        return f"rtsp://{self.__username}:{self.__password}@{self.ip}:{self.port}/Streaming/Channels/{self.channel}"

    def set_password(self, password: str):
        self.__password = password
        return self

    def set_username(self, password: str):
        self.__password = password
        return self

    @staticmethod
    def worker(item):
        cam = Camera(
            ip=item['ip'],
            port=item['port'],
            channel=item['channel'],
            username=item['username'],
            password=item['password']
        )

        while not cam.has_stopped:
            cam.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.stop()
            time.sleep(0.001)

    def stop(self):
        self.has_stopped = True
        self.__capture.release()
        cv2.destroyWindow(self.ip)

    def detect_humans(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Usually frame from VideoCapture is BGR
        boxes, weights = self.__hog.detectMultiScale(gray, winStride=(8, 8))
        return boxes, weights

    #TODO - Convert it to hls to allow streaming in web/mobile
    def read(self):
        ret, frame = self.__capture.read()

        if not ret:
            self.__capture.release()
            time.sleep(1)
            self.__capture = cv2.VideoCapture(self.__rtsp_url, cv2.CAP_FFMPEG)
            return

        frame = cv2.resize(frame, (640, 480))
        boxes, weights = self.throttle.call(self.detect_humans, frame)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow(self.ip, frame)

