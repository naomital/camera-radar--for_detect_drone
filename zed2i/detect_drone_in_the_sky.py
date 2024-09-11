import cv2
import numpy as np


def iou_between_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)
    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_union = area_box1 + area_box2 - area_inter
    if area_union == 0:
        return 0
    return area_inter / area_union


def merge_bounding_boxes(bboxes, iou_threshold=0.05):
    merged_boxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        boxes_to_merge = [current_box]
        non_overlapping_boxes = []
        for box in bboxes:
            if iou_between_boxes(current_box, box) > iou_threshold:
                boxes_to_merge.append(box)
            else:
                non_overlapping_boxes.append(box)
        x_min = min([box[0] for box in boxes_to_merge])
        y_min = min([box[1] for box in boxes_to_merge])
        x_max = max([box[0] + box[2] for box in boxes_to_merge])
        y_max = max([box[1] + box[3] for box in boxes_to_merge])
        merged_box = (x_min, y_min, x_max - x_min, y_max - y_min)
        merged_boxes.append(merged_box)
        bboxes = non_overlapping_boxes
    return merged_boxes


def detect_drone_in_the_sky(frame,
                            prev_gray,
                            lk_params,
                            fgbg,
                            threshold_contour_rea: int = 1,
                            motion_threshold: float = 0.5
                            )\
        -> (np.ndarray, np.ndarray):

    frame_gry = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fgmask = fgbg.apply(frame_gry)

    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_current_frame = []
    for contour in contours:
        if cv2.contourArea(contour) >= threshold_contour_rea:
            x, y, w, h = cv2.boundingRect(contour)
            bbox_current_frame.append((x, y, w, h))

    merged_boxes = merge_bounding_boxes(bbox_current_frame)

    # Optical flow calculation (if we have a previous frame)
    if prev_gray is not None:
        # Calculate optical flow using Lucas-Kanade method
        # Get the center points of the bounding boxes
        p0 = np.array([[x + w // 2, y + h // 2] for x, y, w, h in merged_boxes], dtype=np.float32)
        p0 = p0.reshape(-1, 1, 2)
        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gry, p0, None, **lk_params)

        # Calculate motion vectors
        motion_vectors = p1 - p0
        motion_magnitudes = np.linalg.norm(motion_vectors, axis=2)

        # Filter based on motion magnitude
        for i, (box, magnitude) in enumerate(zip(merged_boxes, motion_magnitudes)):
            if magnitude > motion_threshold:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "moving", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, frame_gry


def main(video_path: str):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None  # Initialize the previous frame for optical flow
    # Optical flow parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, prev_gray = detect_drone_in_the_sky(frame, prev_gray, lk_params, fgbg)
        cv2.imshow('Drone Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the video
    video_path = r'C:\Users\naomi\Desktop\New folder\DJI_20240902164559_0004_D_001.mp4'
    main(video_path)
