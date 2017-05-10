import numpy as np
import cv2
import os.path
import os
import pickle
import copy
import freeman
import l2_dist
import img
import sys

database = []

if sys.argv[1] == '-dump':
    PATH = 'database/'
    LIMIT = -1
    files = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f)) and '.pgm' in f]

    print('Reading files...')
    for f in files:
        tokens = f.split('-')
        class_name = tokens[0]
        nb = int(tokens[1].split('.')[0])

        im, scales = img.normalize(cv2.imread(PATH + f, 0))

        _, im = cv2.threshold(
            im,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        print('Adding {} to database'.format(f))
        database.append([(im, scales), f, class_name])

    print('Rescaling database...')
    scale_x = min(map(lambda lab : lab[0][1][0], database))
    scale_y = min(map(lambda lab : lab[0][1][1], database))

    for lab in database:
        lab[0] = cv2.resize(
            lab[0][0],
            None,
            fx = scale_x / lab[0][1][0],
            fy = scale_y / lab[0][1][1],
            interpolation = cv2.INTER_AREA
        )

    pickle.dump((database, scale_x, scale_y), open('db_dump.pkl', 'wb'))

elif sys.argv[1] == '-classify':
    database, scale_x, scale_y = pickle.load(open('db_dump.pkl', 'rb'))

    unlab = img.normalize(cv2.imread(sys.argv[2], 0))

    if scale_x >= unlab[1][0]:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    unlab = cv2.resize(
        unlab[0],
        None,
        fx = scale_x / unlab[1][0],
        fy = scale_y / unlab[1][1],
        interpolation = interpolation
    )

    h, w = unlab.shape[:2]
    unlab_rotated = img.change_basis(unlab, (w // 2, h // 2), 180)

    _, unlab = cv2.threshold(
        unlab,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, unlab_rotated = cv2.threshold(
        unlab_rotated,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    id_per_class = { }
    for i,(_,_,n) in enumerate(database):
        if not n in id_per_class:
            id_per_class[n] = []
        id_per_class[n].append(i)

    database.append((unlab, sys.argv[2], ''))
    database.append((unlab_rotated, sys.argv[2], ''))
    dist = freeman.Freeman(database)

    f_class = open('classes.csv', 'r')
    id_of_class_raw = f_class.read().split("\n")
    id_of_class = { }
    class_of_id = { }

    for (i,r) in enumerate(id_of_class_raw):
        id_of_class[r.split(',')[0]] = i
        class_of_id[i] = r.split(',')[0]

    max_d = 0
    to_output = []
    for class_name, indices in id_per_class.items():
        d = min([dist.dist(-2, i) for i in indices])
        d_rotated = min([dist.dist(-1, i) for i in indices])
        d = min(d, d_rotated)
        max_d = max(max_d,d)
        to_output.append((id_of_class[class_name],d))

    to_output.sort(key=lambda x: x[0])

    for id_class, d in to_output:
        print('%.2f'%(1-d/max_d))
