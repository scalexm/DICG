import numpy as np
import math
from PIL import Image

def extract_pixels(im, treshold = 0):
    (w, h) = im.size
    pixels = []

    for x in range(w):
        for y in range(h):
            if im.getpixel((x, y)) > treshold:
                pixels.append([x, y])
    return np.array(pixels).T

def pca(pixels):
    (values, vectors) = np.linalg.eig(np.cov(pixels))

    if values[1] > values[0]:
        values[0], values[1] = values[1], values[0]
        vectors = vectors[::-1,:]

    angle = math.atan2(vectors[0][1], vectors[0][0]) * 180 / math.pi
    scales = [int(math.sqrt(values[0])), int(math.sqrt(values[1]))]

    return (angle, scales)

def get_center(pixels):
    x = np.sum(pixels[0, : ]) / pixels.shape[1]
    y = np.sum(pixels[1, : ]) / pixels.shape[1]

    return (x, y)

def change_basis(im):
    (w, h) = im.size
    pixels = extract_pixels(im)
    (angle, scales) = pca(pixels)
    (center_x, center_y) = get_center(pixels)

    new_im = im.rotate(
        (-1) * angle,
        expand = True,
        center = (center_x, center_y),
        translate = (w / 2 - center_x, h / 2 - center_y)
    )

    return (new_im, scales)

def mutual_rescale(imm1, scale1, imm2, scale2):
    scale_x = max(scale1[0], scale2[0])
    scale_y = max(scale1[1], scale2[1])

    (w1, h1) = imm1.size
    (w2, h2) = imm2.size

    w1 *= scale_x / scale1[0]
    h1 *= scale_y / scale1[1]
    w2 *= scale_x / scale2[0]
    h2 *= scale_y / scale2[1]

    im1 = imm1.resize((int(w1), int(h1)))
    im2 = imm2.resize((int(w2), int(h2)))

    (w1, h1) = im1.size
    (w2, h2) = im2.size

    (w, h) = (max(w1, w2), max(h1, h2))
    region1 = (int(w / 2 - w1 / 2), int(h / 2 - h1 / 2))
    region2 = (int(w / 2 - w2 / 2), int(h / 2 - h2 / 2))

    new_im1 = Image.new('L', (w, h))
    new_im2 = Image.new('L', (w, h))

    new_im1.paste(im1, region1)
    new_im2.paste(im2, region2)

    return (new_im1, new_im2)

im1 = Image.open('database/brick-1.pgm')
(im1, s1) = change_basis(im1)

im2 = Image.open('database/bat-7.pgm')
(im2, s2) = change_basis(im2)

(im1, im2) = mutual_rescale(im1, s1, im2, s2)

im1.show()
im2.show()
