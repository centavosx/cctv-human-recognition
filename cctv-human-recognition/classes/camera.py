import time
import os
import cv2
from .throttle import Throttle
from .tensor import Tensor
import numpy as np
import subprocess

class Camera:
    height = 480
    width = 640

    def __init__(self, ip: str, port: str, username: str, password: str, channel: str):
        output_dir = f'output/{ip}'
        os.makedirs(output_dir, exist_ok=True)
        self.ip = ip
        self.port = port
        self.__username = username
        self.__password = password
        self.channel = channel
        self.__capture = cv2.VideoCapture(self.__rtsp_url)
        self.__hog = cv2.HOGDescriptor()
        self.has_stopped = False
        self.__hog.setSVMDetector(self.__hog.getDefaultPeopleDetector())
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        self.activity_threshold = 0.5
        self.ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', '25',
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-f', 'hls',
            '-hls_time', '0.5',
            '-hls_list_size', '3',
            '-hls_flags', 'delete_segments+append_list+omit_endlist',
            os.path.join(output_dir, 'stream.m3u8')
        ], stdin=subprocess.PIPE)

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
        tensor = Tensor("graph.pb")
        cam = Camera(
            ip=item['ip'],
            port=item['port'],
            channel=item['channel'],
            username=item['username'],
            password=item['password']
        )

        current_throttle = Throttle(interval_secs=0.5)

        while not cam.has_stopped:
            frame = cam.read()

            if frame is not None:
                if tensor is None:
                    boxes, weight = current_throttle.call(cam.detect_human, frame=frame)
                    cam.set_boxes_and_write(boxes=boxes, frame=frame)
                else:
                    output_dict = current_throttle.call(cam.detect_human_activity, frame=frame, tensor=tensor)
                    cam.set_human_activity_boxes_and_write(output_dict=output_dict, frame=frame, tensor=tensor)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.stop()

        if tensor is not None:
            tensor.detection_graph_session.close()

    def stop(self):
        self.has_stopped = True
        self.__capture.release()
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()

    def detect_human(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Usually frame from VideoCapture is BGR
        boxes, weights = self.__hog.detectMultiScale(gray, winStride=(8, 8))

        return boxes, weights

    def detect_human_activity(self, frame, tensor: 'Tensor'):
        frame_exp = np.expand_dims(frame, axis=0)
        output_dict = tensor.detection_graph_session.run(tensor.tensor_dict, feed_dict={tensor.image_tensor: frame_exp})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

    def set_human_activity_boxes_and_write(self, frame, output_dict, tensor: 'Tensor'):
        h, w, _ = frame.shape

        for i in range(output_dict['num_detections']):
            cls = int(output_dict['detection_classes'][i])
            score = output_dict['detection_scores'][i]

            # Skip unwanted classes
            if cls in [1, 3, 17, 37, 43, 45, 46, 47, 59, 65, 74, 77, 78, 79, 80]:
                continue
            if score < self.activity_threshold:
                continue

            bbox = output_dict['detection_boxes'][i]

            y_min = int(bbox[0] * self.height)
            x_min = int(bbox[1] * self.width)
            y_max = int(bbox[2] * self.height)
            x_max = int(bbox[3] * self.width)

            idx = cls - 1
            color = self.colors[idx % len(self.colors)]
            label = tensor.model_labels[idx] if idx < len(tensor.model_labels) else f"Class {cls}"

            try:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, max(y_min - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"[WARN] Drawing failed for class {cls}: {e}")

        self.ffmpeg_process.stdin.write(frame.tobytes())

    def set_boxes_and_write(self, frame, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.ffmpeg_process.stdin.write(frame.tobytes())

    def read(self):
        ret, frame = self.__capture.read()

        if not ret:
            self.__capture.release()
            time.sleep(1)
            self.__capture = cv2.VideoCapture(self.__rtsp_url)
            return None

        return cv2.resize(frame, (self.width, self.height))


