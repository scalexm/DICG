import numpy as np
import math
import cv2
import os.path
import os
import pickle
import copy
import freeman

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

# db: idem que le global database
# i_query: indice de la requete dans db
# reference_set: sous ensemble des indices de db reservés au train
def classify(db, i_query, reference_set, dist_class, K = 1):
    dist = [(dist_class.dist(i_db,i_query), db[i_db][2]) for i_db in reference_set]
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
LIMIT = -1
files = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f)) and '.pgm' in f]
files = files[ : LIMIT]

database = []

#########################

if not os.path.isfile("db_dump.pkl"):
    print('Reading files...')
    for f in files:
        tokens = f.split('-')
        class_name = tokens[0]
        nb = int(tokens[1].split('.')[0])

        features = normalize(cv2.imread(PATH + f, 0))

    
        print('Adding {} to database'.format(f))
        database.append([features, f, class_name])

    print('Rescaling database...')
    scale_x = max(map(lambda lab : lab[0][1][0], database))
    scale_y = max(map(lambda lab : lab[0][1][1], database))

    for lab in database:
        lab[0] = cv2.resize(lab[0][0], None, fx = scale_x / lab[0][1][0], fy = scale_y / lab[0][1][1])

    pickle.dump((database,scale_x,scale_y), open("db_dump.pkl", "wb"))
else:
    print("Loading dump...")
    database,scale_x,scale_y = pickle.load(open("db_dump.pkl","rb"))

#######################
train_per_class = 0.7
print("Splitting the database ({}% in train)...".format(train_per_class*100))

# list of id's in class 'name'
id_per_class = {}
for i,(_,_,n) in enumerate(database):
    if not n in id_per_class:
        id_per_class[n] = []
    id_per_class[n].append(i)

train_set = []
test_set = []
for class_name in id_per_class:
    nb_in_class = len(id_per_class[class_name])
    choice_in_class = list(np.random.choice(id_per_class[class_name],int(nb_in_class*train_per_class),replace=False))
    train_set += choice_in_class
    test_set += list(set(id_per_class[class_name])-set(choice_in_class))

#######################
print('Preprocess freeman med')
freeman_med_dist = freeman.freeman_median(database,train_set)

print('Classifying freeman...')
count = 0
db_count = { }

class_seen = {}
elague_train_set = []
for i in train_set:
    the_class = database[i][2]
    if not the_class in class_seen:
        elague_train_set.append(i)
        class_seen[the_class] = True

for i_t in test_set:
    t = database[i_t]
    if t[2] not in db_count:
        db_count[t[2]] = [0, 1]
    else:
        db_count[t[2]][1] += 1

    #img = cv2.resize(t[0][0], None, fx = scale_x / t[0][1][0], fy = scale_y / t[0][1][1])
    #img = t[0]
    label = classify(database, i_t, elague_train_set, freeman_med_dist)
    print('{} : {}'.format(t[1], label))

    if label == t[2]:
        count += 1
        db_count[t[2]][0] += 1


print('Global rate: {}'.format(count / len(test_set)))
for k, v in db_count.items():
    print('`{}` rate: {}/{}'.format(k, v[0], v[1]))


'''
#######################
print('Classifying L2...')
count = 0
db_count = { }

for i_t in test_set:
    t = database[i_t]
    if t[2] not in db_count:
        db_count[t[2]] = [0, 1]
    else:
        db_count[t[2]][1] += 1

    #img = cv2.resize(t[0][0], None, fx = scale_x / t[0][1][0], fy = scale_y / t[0][1][1])
    img = t[0]
    label = classify(img, [database[j] for j in train_set], L2)
    print('{} : {}'.format(t[1], label))

    if label == t[2]:
        count += 1
        db_count[t[2]][0] += 1


print('Global rate: {}'.format(count / len(test_set)))
for k, v in db_count.items():
    print('`{}` rate: {}/{}'.format(k, v[0], v[1]))
'''