import numpy as np
import math
import cv2

def extract_pixels(img):
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = img.shape[:2]
    pixels = [[x, y] for y in range(h) for x in range(w) if img[y, x] > 0]
    return np.array(pixels).T

def get_pc(pixels):
    values, vectors = np.linalg.eig(np.cov(pixels))
    i = 0 if values[0] > values[1] else 1

    angle = math.atan2(vectors[i][1], vectors[i][0]) * 180 / math.pi
    scales = [math.sqrt(values[i]), math.sqrt(values[1 - i])]

    if angle < 0:
        angle += 180
    return (angle, scales)

def get_center(pixels):
    x = np.sum(pixels[0, :]) / pixels.shape[1]
    y = np.sum(pixels[1, :]) / pixels.shape[1]
    return (x, y)

def change_basis(img, center, angle):
    h, w = img.shape[:2]
    c_x, c_y = center

    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    w = int((h * sin) + (w * cos))
    h = int((h * cos) + (w * sin))
    M[0, 2] += (w / 2) - c_x
    M[1, 2] += (h / 2) - c_y

    return cv2.warpAffine(img, M, (w, h))

def normalize(img):
    pixels = extract_pixels(img)
    angle, scales = get_pc(pixels)
    center = get_center(pixels)
    return (change_basis(img, center, -angle), scales)
