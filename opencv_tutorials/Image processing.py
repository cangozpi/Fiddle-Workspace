import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt


def obtain_sample_img():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("could not open camera.")
        sys.exit()
    ret, frame = cap.read()
    if not ret:
        print("could not capture frame from camera.")
        sys.exit()
    cap.release()
    return frame


def colorspaces_functionality():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("frame could not be captured")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame cannot be read.")
            sys.exit()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # threshold the HSV image to get only blue colors in range
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        res = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        cv.imshow('res', res)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()


def geometric_functionality():
    img = obtain_sample_img()
    height, width = img.shape[:2]
    res = cv.resize(img, (width // 10, height // 10),
                    interpolation=cv.INTER_AREA)

    cv.imshow('Resize', res)

    M = np.float32([
        [1, 0, 100],
        [0, 1, 50]
    ])

    dst = cv.warpAffine(img, M, (width, height))

    cv.imshow('translation', dst)

    # Translation
    cv.waitKey(0)

    cv.destroyAllWindows()


def thresholding_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    cv.imshow('THRESH_BINARY', thresh1)

    thresh2 = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    cv.imshow('ADAPTIVE_THRESH_MEAN_C', thresh2)

    ret3, thresh3 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    cv.imshow("THRESH_OTSU", thresh3)

    cv.waitKey(0)
    cv.destroyAllWindows()


def smoothing_functionality():
    img = obtain_sample_img()

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv.filter2D(img, -1, kernel)
    cv.imshow('image', dst)

    dst2 = cv.blur(img, (5, 5))
    cv.imshow("BLUR", dst2)

    dst3 = cv.medianBlur(img, 5)
    cv.imshow("MEDIAN BLUR", dst3)

    dst4 = cv.bilateralFilter(img, 9, 75, 75)
    cv.imshow("BILATERAL FILTERING", dst4)

    cv.waitKey(0)
    cv.destroyAllWindows()


def morphological_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imshow('img', th)

    # kernel = np.ones((5, 5), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    erosion = cv.erode(th, kernel, iterations=1)
    cv.imshow("erosion", erosion)

    dilation = cv.dilate(erosion, kernel, iterations=1)
    cv.imshow("dilation", dilation)

    # erosion + dilation
    opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    cv.imshow("OPENING", opening)

    cv.waitKey(0)
    cv.destroyAllWindows()


def gradient_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    laplacian = cv.Laplacian(gray, cv.CV_64F)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    cv.imshow("laplacian", laplacian)
    cv.imshow("sobelx", sobelx)

    cv.waitKey(0)
    cv.destroyAllWindows()


def edge_detection_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 100, 200)
    cv.imshow("Canny Edge Detection", edges)

    cv.waitKey(0)
    cv.destroyAllWindows()


def image_pyramid_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    lower_reso = cv.pyrDown(gray)
    cv.imshow("lower_reso", lower_reso)

    higher_reso = cv.pyrUp(gray)
    cv.imshow("higher_reso", higher_reso)

    cv.waitKey(0)
    cv.destroyAllWindows()


def countours_functionality():
    img = obtain_sample_img()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(gray, contours, -1, (0, 255, 0), 3)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # colorspaces_functionality()
    # geometric_functionality()
    # thresholding_functionality()
    # smoothing_functionality()
    # morphological_functionality()
    # gradient_functionality()
    # edge_detection_functionality()
    # image_pyramid_functionality()
    countours_functionality()
