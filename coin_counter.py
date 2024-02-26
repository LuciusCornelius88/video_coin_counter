import cv2
import cvzone
import shutil
import numpy as np
from pathlib import Path


def delete_cache(input_path):
    for path in input_path.iterdir():
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
        elif path.is_dir():
            delete_cache(path)


def nothing(x):
    ...


def set_trackbars(window_name):
    cv2.createTrackbar('canny_thres_min', window_name, 170, 255, nothing)
    cv2.createTrackbar('canny_thres_max', window_name, 255, 255, nothing)

    cv2.createTrackbar('d', window_name, 5, 20, nothing)
    cv2.createTrackbar('sigmaColor', window_name, 80, 150, nothing)
    cv2.createTrackbar('sigmaSpace', window_name, 80, 150, nothing)


def blur_frame(frame, d=20, sigma_color=80, sigma_space=80):
    frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    return frame


def process_frame(frame, canny_thres_min=200, canny_thres_max=255):
    frame = cv2.Canny(frame, canny_thres_min, canny_thres_max)
    kernel = np.ones((3, 3), np.uint8)

    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    return frame


def process_contours(frame, contours, total_money):
    for contour in contours:
        peri = cv2.arcLength(contour['cnt'], True)
        approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

        if len(approx) > 5:
            area = contour['area']
            x, y, w, h = contour['bbox']

            if area >= 3000:
                val = 10
                total_money += val
                cv2.putText(frame, str(val), (x + w + 3, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif 2300 <= area < 3000:
                val = 5
                total_money += val
                cv2.putText(frame, str(val), (x + w + 3, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif 1500 <= area < 2300:
                val = 2
                total_money += val
                cv2.putText(frame, str(val), (x + w + 3, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif 500 < area < 1500:
                val = 1
                total_money += val
                cv2.putText(frame, str(val), (x + w + 3, y + h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, total_money


def main():
    window_name = 'Window'

    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    set_trackbars(window_name)

    while True:
        total_money = 0
        image_count = np.zeros((480, 640, 3), np.uint8)

        ret, frame = cap.read()

        if not ret:
            break

        canny_thres_min = cv2.getTrackbarPos('canny_thres_min', window_name)
        canny_thres_max = cv2.getTrackbarPos('canny_thres_max', window_name)

        d = cv2.getTrackbarPos('d', window_name)
        sigma_color = cv2.getTrackbarPos('sigmaColor', window_name)
        sigma_space = cv2.getTrackbarPos('sigmaSpace', window_name)

        processed_frame = blur_frame(frame, d, sigma_color, sigma_space)
        processed_frame = process_frame(frame, canny_thres_min, canny_thres_max)

        frame, contours = cvzone.findContours(frame, processed_frame, minArea=20)

        if contours:
            frame, total_money = process_contours(frame, contours, total_money)

        cvzone.putTextRect(image_count, f'Rs.{total_money}', (100, 200), scale=10, offset=30, thickness=7)
        image_stacked = cvzone.stackImages([frame, processed_frame, image_count], 2, 1)

        cv2.imshow('img', image_stacked)

        if cv2.waitKey(1) == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    delete_cache(Path(__file__).parent)
