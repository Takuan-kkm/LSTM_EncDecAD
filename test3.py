import cv2
import sys
import pickle
import numpy as np

# Load score
with open("ascore_confuse.pkl", "rb") as f:
    score = pickle.load(f)

vid = cv2.VideoCapture("video/confuse10fps.mp4")
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 360))

delay = 10
window_name = 'frame'
font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC
np.set_printoptions(precision=1)

i = 0

if not vid.isOpened():
    sys.exit()

while True:
    ret, frame = vid.read()
    i += 1
    if i < 5:
        continue
    if i > 2097:
        break

    if ret:
        if score[i - 6] > 150:
            color = (30, 30, 255)
        else:
            color = (0, 255, 0)
        # time
        cv2.rectangle(frame, (0, 0), (150, 35), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "t:" + str(i / 10), (15, 30), font, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)

        # ascore
        cv2.rectangle(frame, (0, 325), (640, 360), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (320, 325), (320 + int(score[i - 5]), 360), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (320 - int(score[i - 5]), 325), (320, 360), color, -1, cv2.LINE_AA)
        cv2.putText(frame, "ascore:" + str(round(score[i - 5], 2)), (200, 350), font, 1, (0, 0, 0), 2, cv2.LINE_AA)  # (B,G,R)

        cv2.imshow(window_name, frame)
        out.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

vid.release()
out.release()
cv2.destroyWindow(window_name)
