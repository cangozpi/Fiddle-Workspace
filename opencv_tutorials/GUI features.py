import cv2 as cv
import sys
import numpy as np


def image_functionality():
    img = cv.imread(cv.samples.findFile("starry_night.jpg"))

    if img is None:
        sys.exit("Could not read the image.")

    cv.imshow("Display Window", img)

    k = cv.waitKey(0)

    if k == ord("s"):
        cv.imwrite("starry_night.png", img)


def video_functionality():
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera cannot be opened.")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("frame cannot be received.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        out.write(gray)
        cv.imshow('live feed', gray)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


def drawing_functionality():
    img = np.zeros((512, 512, 3), np.uint8)

    frame = cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, 'OpenCV', (10, 500), font,
               4, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow('drawn image', frame)
    k = cv.waitKey(0)
    if k == ord('q'):
        sys.exit()


def mouse_functionality():
    # mouse callback
    def draw_circle(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(img, (x, y), 100, (255, 0, 0), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()


def mouse_functionality_advanced():
    global drawing, mode, ix, iy
    drawing = False  # true if mouse is pressed
    mode = True  # if True draw rectange. Press 'm' to toggle to curve
    ix, iy = -1, -1

    # mouse callback
    def draw_circle(event, x, y, flags, param):
        global drawing, mode, ix, iy

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
                else:
                    cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)

    while True:
        cv.imshow('image', img)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break

    cv.destroyAllWindows()


def trackbar_functionality():
    def nothing(x):
        pass

    img = np.zeros((300, 512, 3), np.uint8)
    cv.namedWindow('image')

    # trackbars for color change
    cv.createTrackbar('R', 'image', 0, 255, nothing)
    cv.createTrackbar('G', 'image', 0, 255, nothing)
    cv.createTrackbar('B', 'image', 0, 255, nothing)

    # switch for ON/OFF
    switch = '0 : OFF \n1: ON'
    cv.createTrackbar(switch, 'image', 0, 1, nothing)

    while True:
        cv.imshow('image', img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("q"):
            break

        # current positions of trackbars
        r = cv.getTrackbarPos('R', 'image')
        g = cv.getTrackbarPos('G', 'image')
        b = cv.getTrackbarPos('B', 'image')
        s = cv.getTrackbarPos(switch, 'image')

        if s == 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv.destroyAllWindows()
    sys.exit()


if __name__ == "__main__":
    # Uncomment functions below 1 by 1 to play with demos:

    # image_functionality()
    # video_functionality()
    # drawing_functionality()
    # mouse_functionality()
    # mouse_functionality_advanced()
    # trackbar_functionality()
