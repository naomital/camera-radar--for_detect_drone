import cv2
import numpy as np

# Initialize variables
tracking = False
init_point = None
old_points = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Mouse callback function
def select_point(event, x, y, flags, param):
    global tracking, init_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        init_point = (x, y)
        old_points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        tracking = True


# Create a VideoCapture object
cap = cv2.VideoCapture(r"C:\Users\naomi\Desktop\New folder\DJI_20240902164559_0004_D_002.mp4")
cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)

# Create a window and set a mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

        # Draw the tracking point
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

        old_gray = frame_gray.copy()
        old_points = new_points.reshape(-1, 1, 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()