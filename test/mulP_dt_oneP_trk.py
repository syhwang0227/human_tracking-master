# Human Detection + Pose Estimation for one person

import os
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf_v2

from PIL import Image, ImageFilter

import logging
import traceback

tf_v1 = tf_v2.compat.v1
# Eager Mode: False
tf_v1.disable_v2_behavior()
tf_v1.disable_eager_execution()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    filename='test01.log',
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
        height, width = image.shape[0], image.shape[1]
        for idx in range(len(bbox_list)):
            bbox = bbox_list[idx]
            label = label_list[idx]
            conf = conf_list[idx]

            # draw bbox

            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            # cv2.circle(image, (center_x, center_y), 5, (255, 0, 255), cv2.FILLED)  # 중심점

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            # put label text
            cv2.putText(image, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))

    # def img_crop(img, bbox_cd):
    #     # bbox 영역 자르기
    #     bbox_cd = []
    #     for i, (x1, y1, x2, y2) in enumerate(bbox_list):
    #         bbox_cd = [x1, y1, x2, y2]
    #         print(bbox_cd)
    #
    #         result_crop = []
    #         for j in bbox_cd:
    #             bbox_crop = img_copy[x1:x2, y1:y2]
    #             result_crop.append(bbox_crop)
    #
    #     return result_crop


if __name__ == '__main__':
    humanDetector = HumanDetector().run()

    cap = cv2.VideoCapture('../test.avi')

    # 사용할 테스트 동영상의 크기 구하기
    # frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print('frame_size', frame_size)

    # save video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('one_detection_bbox.avi', fourcc, 30.0, (1024, 768))  # 계속 재생 에러가 발생했던 이유: 원본 동영상과 크기가 달라서

    try:
        # Setup mediapipe instance
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
                        bbox_crop = inference_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # image crop / 원본 이미지를 사용하는 이유: visualize_img는 bbox가 그려져있다.
                        # cv2.imshow('{}_img'.format(i), bbox_crop)
                        cv2.imwrite("test/crop_img.jpg", bbox_crop)
                        bbox_imgs.append(bbox_crop)
                        # print(bbox_imgs)
                        # for j in range(len(bbox_imgs)):
                        #     imgs = bbox_imgs[j]
                        #     imgs.save('images_{}.jpg'.format(j))

                    # numpy_array = np.array(bbox_imgs)
                    # img2 = Image.fromarray(numpy_array, "RGB")
                    # img2.show()

                    # for j in range(len(bbox_imgs)):
                    #     imgs = bbox_imgs[j]
                    #     imgs.save('images_{}.jpg'.format(j))

                    # cv2.imshow('Mediapipe', cv2.resize(bbox_imgs, (640, 480)))

                    # Recolor image to RGB
                    visualize_img = cv2.cvtColor(visualize_img, cv2.COLOR_BGR2RGB)
                    visualize_img.flags.writeable = False

                    # Make detection
                    results = pose.process(visualize_img)
                    # print(results)

                    # Recolor back to BGR
                    visualize_img.flags.writeable = True
                    visualize_img = cv2.cvtColor(visualize_img, cv2.COLOR_RGB2BGR)

                    # Render detections
                    mp_drawing.draw_landmarks(visualize_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                                              # mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              # mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    out.write(visualize_img)
                    cv2.imshow('Mediapipe', cv2.resize(visualize_img, (640, 480)))
                        # Multi-person에 대한 값 출력하기
                        # Try 1. 가장 단순하게 접근 - cv2.add 사용
                        # 문제점1: 실행 후 cv2.imshow() 작동 전에 종료 / 에러 표시 없음 / 이게뭐람;
                        # 해결1: 들여쓰기...?
                        # 문제점2: 영상이 출력은 되나 여전히 한 명의 사람만 Pose Estimation이 적용된다.
                        # 내가 생각한 문제점의 원인: bbox_crop의 Pose Estimation 정보를 가져오지 않고 단순히 영상만 가져와서 visualize_img에 합쳐진 것 같다.
                        # 그래서 이 문제는 이미지나 동영상 합치기가 아닌 정보 값을 가져와야 하는 문제이다.
                        # 동영상 단위로 합친다면 사람이 디텍션 될 때 마다 저장하고 붙이는 작업을 해야 하고 결과적으로 엄청난 속도가 걸릴 것이다.
                        # result_mp = cv2.add(visualize_img, bbox_crop)
                        # cv2.imshow('result', result_mp)

                        # Try 2. Keypoints에 대한 값 가져와서 visualize_img에 표시하기



                else:
                    print('Error')
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    except:
        logging.error(traceback.format_exc())

    cap.release()
    out.release()
    cv2.destroyAllWindows()
