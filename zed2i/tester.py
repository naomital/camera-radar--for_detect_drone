
import cv2
import numpy as np
import pyzed.sl as sl

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.VGA
cam = sl.Camera()
mat = sl.Mat()
err = cam.open(init)
cam.retrieve_image(mat, sl.VIEW.RIGHT)
frame_r = mat.get_data().copy()
frame_bgr = cv2.cvtColor(frame_r, cv2.COLOR_RGBA2BGR)
print(frame_r.shape)
size = (frame_r.shape[1], frame_r.shape[0])  # (width, height)
result = cv2.VideoWriter('test_output.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         20, size)

if not result.isOpened():
    print("Error: VideoWriter not opened!")
else:
    for _ in range(100):
        # Create a dummy frame (a solid green image)
        frame = np.zeros((376, 672, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)  # Green color
        result.write(frame_bgr)

    result.release()
    print("Video saved successfully as 'test_output.mp4'")
