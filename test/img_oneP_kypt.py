# 한 장의 이미지에서 한 명에 대한 keypoint 출력

import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image = cv2.imread("test_img.png")

    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 포맷
    image2.flags.writeable = False

    result_img = pose.process(image2)
    # print(result_img)

    image2.flags.writeable = True
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)  # BGR 포맷

    mp_drawing.draw_landmarks(image2, result_img.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # print("landmark: ", result_img.pose_landmarks)
    # print("pose_connection: ", mp_pose.POSE_CONNECTIONS)

    cv2.imshow("image", image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()