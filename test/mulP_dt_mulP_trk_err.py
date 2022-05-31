# Landmark Connection 까지 완성된 코드 (21.09.14 기준)

import os
import json
from pprint import pprint

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf_v2

import logging
import traceback

tf_v1 = tf_v2.compat.v1
# Eager Mode: False
tf_v1.disable_v2_behavior()
tf_v1.disable_eager_execution()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

mp_pose.POSE_CONNECTIONS = [[0, 4], [4, 5], [5, 6], [6, 8], [0, 1],
                            [1, 2], [2, 3], [3, 7], [10, 9], [12, 11],
                            [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
                            [18, 20], [11, 13], [13, 15], [15, 17], [15, 19],
                            [15, 21], [17, 19], [12, 14], [11, 23], [24, 13],
                            [24, 26], [23, 25], [26, 28], [25, 27], [28, 30],
                            [27, 29], [30, 32], [29, 31], [28, 32], [25, 31]]

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename='./test01.log',
    level=logging.ERROR)


class HumanDetector:

    def __init__(self,
                 model_file=os.path.join('/home/di-05/Downloads/hsy/human_tracking/human_detection_model.pb'),
                 label_file=os.path.join('/home/di-05/Downloads/hsy/human_tracking/human_label_map.json'),
                 threshold=0.5):
        self.model_file = model_file
        self.label_file = label_file
        self.threshold = threshold
        self.detection_graph = None
        self.default_graph = None
        self.session = None
        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None
        self.label_map = None

    def run(self):
        # Load model
        self.detection_graph = tf_v1.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf_v1.GraphDef()
            with tf_v1.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf_v1.import_graph_def(od_graph_def, name='')
        self.default_graph = self.detection_graph.as_default()
        self.session = tf_v1.Session(graph=self.detection_graph)

        # Load tensor
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Load label map
        self.label_map = json.loads(open(self.label_file, 'r').read())
        return self

    def inference(self, image):
        image_np = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num_detections) = self.session.run(
            fetches=[self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np}
        )
        height, width = image.shape[0], image.shape[1]
        boxes_list = [None for _ in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            x1, y1 = int(boxes[0, i, 1] * width), int(boxes[0, i, 0] * height)
            x2, y2 = int(boxes[0, i, 3] * width), int(boxes[0, i, 2] * height)
            boxes_list[i] = [x1, y1, x2, y2]

        _boxes = boxes_list
        _scores = scores[0].tolist()
        _classes = [int(x) for x in classes[0].tolist()]
        _num_detections = int(num_detections[0])
        return _boxes, _scores, _classes, _num_detections

    def get_data(self, image):
        """
        Args:
            image (numpy.ndarray): Input Image
        Returns:
            bbox_list: Detected Human Bounding Box
            label_list: Detected Human Label
            conf_list: Detected Human Confidence
        """
        boxes, scores, classes, num_detections = self.inference(image)
        bbox_list = list()
        label_list = list()
        conf_list = list()
        for label in self.label_map:
            for idx in range(len(boxes)):
                if classes[idx] == self.label_map[label] and scores[idx] > self.threshold:
                    bbox_list.append(boxes[idx])
                    label_list.append(label)
                    conf_list.append(scores[idx])
        return bbox_list, label_list, conf_list

    @staticmethod
    def draw(image, bbox_list, label_list, conf_list):
        for idx in range(len(bbox_list)):
            bbox = bbox_list[idx]
            label = label_list[idx]
            conf = conf_list[idx]
            # draw bbox
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # print("size:", bbox[0], bbox[1], bbox[2], bbox[3])
            # put label text
            cv2.putText(image, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))


if __name__ == '__main__':
    humanDetector = HumanDetector().run()

    cap = cv2.VideoCapture('../test.avi')

    # save video
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1024, 768))

    try:
        # Setup MediaPipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    inference_img = np.copy(frame)
                    visualize_img = np.copy(frame)

                    # Human Detection
                    bbox_list, label_list, conf_list = humanDetector.get_data(inference_img)
                    # humanDetector.draw(image, bbox_list, label_list, conf_list)
                    humanDetector.draw(visualize_img, bbox_list, label_list, conf_list)
                    # cv2.imshow('test', visualize_img)
                    # cv2.imshow('test', cv2.resize(visualize_img, (640, 480)))

                    # Pose Estimation
                    # 원본에 영향을 주지 않기 위해 이미지 복사
                    # img_copy = inference_img.copy()

                    # bbox 형태로 이미지 자르기
                    bbox_imgs = list()
                    bbox_crop = list()
                    for i in range(len(bbox_list)):
                        bbox = bbox_list[i]
                        bbox_crop = inference_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # image crop
                        # cv2.imshow('{}_img'.format(i), bbox_crop)
                        bbox_imgs.append(bbox_crop)
                        # cv2.imwrite("img/crop_img_{}.jpg".format(i), bbox_crop)

                        # bbox 형태로 잘린 이미지에 Pose Estimation 적용하기
                        bbox_crop = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2RGB)
                        bbox_crop.flags.writeable = False

                        # Make detection
                        results = pose.process(bbox_crop)

                        # Recolor back to BGR
                        bbox_crop.flags.writeable = True
                        bbox_crop = cv2.cvtColor(bbox_crop, cv2.COLOR_RGB2BGR)

                        # Render detections
                        # mp_drawing.draw_landmarks(bbox_crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        #                           # mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        #                           # mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        #                           )

                        # mp_drawing.draw_landmarks(visualize_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        #                           # mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        #                           # mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        #                           )

                        bbox_crop_h, bbox_crop_w, c = bbox_crop.shape

                        if results.pose_landmarks:
                            # mp_drawing.draw_landmarks(visualize_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                            points = {}  # 딕셔너리가 아닌 [](배열), ()(튜플)일 경우 에러 발생
                            for id, lm in enumerate(results.pose_landmarks.landmark):
                                mp_drawing.draw_landmarks(bbox_crop, results.pose_landmarks,
                                                          mp_pose.POSE_CONNECTIONS)
                                h, w, c = visualize_img.shape
                                # print(id, lm)
                                # cx, cy = int(lm.x * w), int(lm.y * h)  # (lm.x * 너비), (lm.y * 높이)
                                cx, cy = int(lm.x * bbox_crop_w), int(lm.y * bbox_crop_h)

                                # print(lm.x, bbox_crop_w, lm.y, bbox_crop_h)

                                # cv2.circle(visualize_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                                cv2.circle(visualize_img, (cx + bbox[0], cy + bbox[1]), 5, (255, 0, 255), cv2.FILLED)
                                # print(id, ":", (cx + bbox[0], cy + bbox[1]))  # 0 : (996, 124) ~ 32 : (970, 431)

                                points[id] = (cx + bbox[0]), (cy + bbox[1])  # [id]가 없을 경우 line이 출력되지 않는다.

                                # points.append(cx + bbox[0], cy + bbox[1])

                                # cv2.circle(visualize_img, (cx + bbox[2], cy + bbox[3]), 5, (255, 0, 0), cv2.FILLED)
                                # cv2.circle(visualize_img, (cx + (bbox[0], bbox[1]), cy +(bbox[2], bbox[3])), 5, (255, 0, 255), cv2.FILLED)

                            # print(lm)  # x, y, z, visibility
                            # print(len(mp_pose.POSE_CONNECTIONS))  # 35
                            # key = mp_pose.POSE_CONNECTIONS
                            # pprint(mp_pose.POSE_CONNECTIONS)
                            # print(type(points))  # <class 'dict'>

                        # Landmark Connection
                            LC = mp_pose.POSE_CONNECTIONS
                            # print(LC)  # [[0, 4], ~ ,[25, 31]]
                            for a, b in LC:
                                # print("value", a, b)  # value 0 4 ~ value 25 31
                                if a not in points or b not in points: continue
                                # print("points:", points)
                                ax, ay = points[a]
                                # print("ax, ay:", ax, ay)
                                bx, by = points[b]
                                # print("bx, by:", bx, by)
                                # cv2.line(visualize_img, (ax, ay), (bx, by), (0, 255, 255), 1)

                        # cv2.imwrite("img/crop_img_{}.jpg".format(i),
                        # bbox_crop)
                        # cv2.imshow('{}_img'.format(i), bbox_crop)

                    # out.write(visualize_img)
                    # cv2.imshow('Mediapipe', cv2.resize(visualize_img, (640, 480)))
                    cv2.imshow('Mediapipe', visualize_img)

                else:
                    print('Error')
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    except:
        logging.error(traceback.format_exc())

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
