import numpy as np
import math
import cv2
from os import *
from os.path import *

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

def hausdorff(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    r1 = img1[(h1 - h) // 2 : (h1 + h) // 2, (w1 - w) // 2 : (w1 + w) // 2]
    r2 = img2[(h2 - h) // 2 : (h2 + h) // 2, (w2 - w) // 2 : (w2 + w) // 2]

    func = np.vectorize(lambda r, x: np.min(cv2.absdiff(int(x), r)))
    func.excluded.add(0)
    A = np.max(func(r1, r2))
    B = np.max(func(r2, r1))
    return max(A, B)

def L2(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    r1 = img1[(h1 - h) // 2 : (h1 + h) // 2, (w1 - w) // 2 : (w1 + w) // 2]
    r2 = img2[(h2 - h) // 2 : (h2 + h) // 2, (w2 - w) // 2 : (w2 + w) // 2]
    return cv2.norm(r1, r2)

def classify(unlab, lab, dist, K = 1):
    dist = [(dist(unlab, lab[0]), lab[1]) for lab in database]
    dist = sorted(dist, key = lambda z: z[0])
    dist = dist[ : K]

    counts = { }
    for z in dist:
        if z[1] not in counts:
            counts[z[1]] = 1
        else:
            counts[z[1]] += 1

    labels = list(counts.items())
    labels = sorted(labels, key = lambda z : z[1])
    labels = list(reversed(labels))

    return labels[0][0]

# img1 = normalize(cv2.imread('database/apple-1.pgm', 0))
# img2 = normalize(cv2.imread('database/apple-2.pgm', 0))
# print(L2(*rescale(img1, img2)))
# exit(0)

PATH = 'database/'
LIMIT = 15
files = [f for f in listdir(PATH) if isfile(join(PATH, f)) and '.pgm' in f]
files = files[ : LIMIT]

database = []
test = []


#########################
print('Reading files...')
for f in files:
    tokens = f.split('-')
    class_name = tokens[0]
    nb = int(tokens[1].split('.')[0])

    features = normalize(cv2.imread(PATH + f, 0))

    if nb % 2 == 0:
        print('Adding {} to database'.format(f))
        database.append([features, class_name])
    else:
        print('Adding {} to test'.format(f))
        test.append([features, f, class_name])


##############################
print('Rescaling database...')
scale_x = max(map(lambda lab : lab[0][1][0], database))
scale_y = max(map(lambda lab : lab[0][1][1], database))

for lab in database:
    lab[0] = cv2.resize(lab[0][0], None, fx = scale_x / lab[0][1][0], fy = scale_y / lab[0][1][1])


#######################
print('Classifying...')

count = 0
db_count = { }

for t in test:
    if t[2] not in db_count:
        db_count[t[2]] = [0, 1]
    else:
        db_count[t[2]][1] += 1

    img = cv2.resize(t[0][0], None, fx = scale_x / t[0][1][0], fy = scale_y / t[0][1][1])
    label = classify(img, database, L2)
    print('{} : {}'.format(t[1], label))

    if label == t[2]:
        count += 1
        db_count[t[2]][0] += 1


print('Global rate: {}'.format(count / len(test)))
for k, v in db_count.items():
    print('`{}` rate: {}/{}'.format(k, v[0], v[1]))
