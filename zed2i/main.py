import cv2
import numpy as np
import pyzed.sl as sl
from detect_drone_in_the_sky import detect_drone_in_the_sky


def main():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    # save video
    cam.retrieve_image(mat, sl.VIEW.LEFT)
    frame_l = mat.get_data().copy()
    cam.retrieve_image(mat, sl.VIEW.RIGHT)
    frame_r = mat.get_data().copy()
    numpy_vertical_concat = np.concatenate((frame_l, frame_r), axis=1)
    frame_bgr = cv2.cvtColor(numpy_vertical_concat, cv2.COLOR_RGBA2BGR)
    # Correct size tuple
    size = (frame_bgr.shape[1], frame_bgr.shape[0])  # (width, height)
    result = cv2.VideoWriter('filename_.mp4',
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             20, size)

    if not result.isOpened():
        print("Error: VideoWriter not opened!")
        cam.close()
        return

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    prev_frame_gry = None
    key = ''

    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            frame_left, prev_frame_gry = detect_drone_in_the_sky(frame=mat.get_data().copy(), prev_gray=prev_frame_gry,
                                                                 lk_params=lk_params, fgbg=fgbg)
            cam.retrieve_image(mat, sl.VIEW.RIGHT)
            frame_right, prev_frame_gry = detect_drone_in_the_sky(frame=mat.get_data().copy(), prev_gray=prev_frame_gry,
                                                                  lk_params=lk_params, fgbg=fgbg)

            numpy_vertical_concat = np.concatenate((frame_left, frame_right), axis=1)
            frame_bgr = cv2.cvtColor(numpy_vertical_concat, cv2.COLOR_RGBA2BGR)
            cv2.imshow('frame', numpy_vertical_concat)
            result.write(frame_bgr)
        else:
            print("Error during capture : ", err)
            break

        key = cv2.waitKey(5)

    cam.close()
    result.release()


if __name__ == "__main__":
    main()
