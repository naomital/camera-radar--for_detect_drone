# import the necessary packages
import numpy as np
import argparse
import cv2

# define a video capture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)

window_name = "OpenCV"

radius = 11

mouse_position = [0, 0]


def mouse_callback(event, x, y, flags, param):
    global mouse_position
    if event == 1:
        mouse_position[0] = x
        mouse_position[1] = y


cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)  # Mouse callback

cv2.namedWindow("red_mask", cv2.WINDOW_AUTOSIZE)


def get_center_of_mask(mask: np.ndarray):
    kernel = np.ones((10, 10), np.uint8)
    dilate = cv2.dilate(mask, kernel)
    # # cv2.imshow('Dilate', dilate)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Find center of mass of mask
    mass_x, mass_y = np.where(dilate >= 255)
    # mass_x and mass_y are the list of x indices and y indices of mass pixels
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)

    if cent_x >= 0 and cent_y >= 0:
        return round(cent_y), round(cent_x)
    return None


def on_saturation_trackbar(val):
    global red_saturation
    red_saturation = val


def on_value_trackbar(val):
    global red_value
    red_value = val


def on_hue_up_trackbar(val):
    global red_hue_up
    red_hue_up = val


def on_hue_low_trackbar(val):
    global red_hue_low
    red_hue_low = val


def main():
    global red_saturation, red_value, red_hue_up, red_hue_low
    red_saturation = 0
    red_value = 0
    red_hue_up = 0
    red_hue_low = 179

    cv2.createTrackbar("Red Saturation", "red_mask", 0, 255, on_saturation_trackbar)
    cv2.createTrackbar("Red Value", "red_mask", 0, 255, on_value_trackbar)
    cv2.createTrackbar("Red Hue up", "red_mask", 0, 179, on_hue_up_trackbar)
    cv2.createTrackbar("Red Hue low", "red_mask", 179, 179, on_hue_low_trackbar)

    while (True):
        cap.set(cv2.CAP_PROP_BRIGHTNESS, -64)
        # Capture the video frame
        # by frame
        ret, frame = cap.read()

        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # images = frame[120:360, 160:480, :]
        image = frame

        # load the images and convert it to grayscale
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        red = image[:, :, 2]
        blue = image[:, :, 0]
        green = image[:, :, 1]

        cv_image = red

        # perform a naive attempt to find the (x, y) coordinates of
        # the area of the images with the largest intensity value
        (minVal, maxVal, minLoc, maxLoc_red) = cv2.minMaxLoc(red)
        cv2.circle(image, maxLoc_red, 5, (0, 0, 255), 2)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_green = np.array([45, 100, 100])
        upper_green = np.array([75, 255, 255])
        # Threshold the HSV images to get only green colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_loc = get_center_of_mask(green_mask)

        if green_loc is not None:
            cv2.circle(image, green_loc, 5, (0, 255, 0), 2)


        cv2.imshow('green_mask', green_mask)

        # lower boundary RED color range values; Hue (0 - 10)
        # red_lower1 = np.array([0, red_saturation, red_value])
        # red_upper1 = np.array([red_hue_up, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        # red_lower = np.array([red_hue_low, red_saturation, red_value])
        # red_upper = np.array([179, 255, 255])

        # red_lower = np.array([140, 175, 165])
        # red_upper = np.array([179, 190, 190])
        #
        # # red_lower_mask = cv2.inRange(images, red_lower1, red_upper1)
        # red_mask = cv2.inRange(images, red_lower, red_upper)
        # red_loc = get_center_of_mask(red_mask)
        #
        # if red_loc is not None:
        #     cv2.circle(images, red_loc, 5, (255, 0, 0), 2)
        #
        # # red_mask = red_lower_mask + red_upper_mask
        #
        # cv2.imshow('red_mask', red_mask)

        # cv2.imshow('image_with_mask', image_with_mask)

        (minVal, maxVal, minLoc, maxLoc_green) = cv2.minMaxLoc(green)
        x, y = maxLoc_green
        # cv2.circle(images, maxLoc_green, 5, (0, 255, 0), 2)

        # cv2.circle(images, mouse_position, 5, (255, 0, 0), 2)
        # display the results of the naive attempt
        cv2.imshow(window_name, image)

        mouse_x, mouse_y = mouse_position

        print(
            F"mouse_hsv: {hsv[mouse_y, mouse_x]}, red: {maxLoc_red}, green: {maxLoc_green}, green_color: {image[y, x]}")

        # # apply a Gaussian blur to the images then find the brightest
        # # region
        # cv_image = cv2.GaussianBlur(cv_image, (radius, radius), 0)
        # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cv_image)
        # images = orig.copy()
        # cv2.circle(images, maxLoc, radius, (255, 0, 0), 2)
        # # display the results of our newly improved method
        # cv2.imshow("Detection", images)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
