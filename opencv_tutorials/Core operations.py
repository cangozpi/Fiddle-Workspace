import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import time


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


def basic_operations_functionality():
    img = obtain_sample_img()
    cv.imshow('image', img)

    px = img[100, 100]
    print(px)
    blue = img[100, 100, 0]
    print(blue)

    img[200:300, 200:300] = img[300:400, 300:400]
    cv.imshow('image', img)

    k = cv.waitKey(0)

    if k == ord("q"):
        sys.exit()


def arithmetic_operations_functionality():
    img1 = obtain_sample_img()
    time.sleep(2)
    img2 = obtain_sample_img()

    dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

    cv.imshow('image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def arithmetic_operations_advanced_functionality():
    img1 = obtain_sample_img()
    time.sleep(2)
    img2 = obtain_sample_img()

    rows, cols, channels = img2.shape

    roi = img1[0:rows, 0:cols]

    # create a mask and inverse mask
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

    img2_fg = cv.bitwise_and(img1_bg, img2, mask=mask)

    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv.imshow("img", img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # basic_operations_functionality()
    # arithmetic_operations_functionality()
    arithmetic_operations_advanced_functionality()
