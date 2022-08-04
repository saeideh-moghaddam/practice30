import cv2
import keyboard
import numpy as np

video_cap = cv2.VideoCapture(0)

frame_width, frame_height = int(video_cap.get(3)), int(video_cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_rate = 20

def skin_Detection(frame):
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 45, 58], dtype = "uint8")
    lower2 = np.array([172, 45, 58], dtype = "uint8")
    upper = np.array([20, 150, 250], dtype = "uint8")
    upper2 = np.array([179, 150, 250], dtype = "uint8")
    skin_mask = cv2.inRange(frame_HSV, lower, upper)
    skin_mask2 = cv2.inRange(frame_HSV, lower2, upper2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin_mask2 = cv2.GaussianBlur(skin_mask2, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    HSV = cv2.dilate(skin_mask, kernel)
    HSV2 = cv2.dilate(skin_mask2, kernel)
    skin1 = cv2.bitwise_and(frame, frame, mask = HSV)
    skin2 = cv2.bitwise_and(frame, frame, mask =HSV2)
    skin = cv2.bitwise_or(skin1,skin2)
    return skin

while True:
    ret, frame = video_cap.read()
    if not ret or keyboard.is_pressed("esc"):
        break
    frame_result = skin_Detection(frame)

    cv2.imshow("skin detection", frame_result)
    cv2.waitKey(1)
    
video_cap.release()
cv2.destroyAllWindows()
