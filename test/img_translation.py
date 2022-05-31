import cv2
import numpy as np

img = cv2.imread('../img/Mediapipe_screenshot_03.09.2021.png')

num_rows, num_cols = img.shape[:2]

translation_matrix = np.float32([[1,0,300],[0,1,110]])
img_translation = cv2.warpAffine(img,translation_matrix,(num_cols,num_rows),cv2.INTER_LINEAR) # 열,행 순서로
# 열(열이 있는 방향은 가로)
# 행(행이 있는 방향은 세로)
# warpAffine : 영상에 적용하기 위한 함수(이동 matrix를)
cv2.imshow('Trans', img_translation)
cv2.waitKey()
cv2.destroyAllWindows()
