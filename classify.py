import numpy as np
import cv2
import os.path
import os
import pickle
import copy
import freeman
import l2_dist
import img

# db: idem que le global database
# i_query: indice de la requete dans db
# test_set: sous ensemble des indices de db reservés au test
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

        features = img.normalize(cv2.imread(PATH + f, 0))

        print('Adding {} to database'.format(f))
        database.append([features, f, class_name])

    print('Rescaling database...')
    scale_x = min(map(lambda lab : lab[0][1][0], database))
    scale_y = min(map(lambda lab : lab[0][1][1], database))

    for lab in database:
        lab[0] = cv2.resize(lab[0][0], None, fx = scale_x / lab[0][1][0], fy = scale_y / lab[0][1][1], interpolation = cv2.INTER_AREA)

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


##########################
print('Classifying...')
count = 0
db_count = { }

L2_clf = freeman.Freeman(database)

for i_t in test_set:
    t = database[i_t]
    if t[2] not in db_count:
        db_count[t[2]] = [0, 1]
    else:
        db_count[t[2]][1] += 1

    img = t[0]
    label = classify(database, i_t, train_set, L2_clf,1)
    print('{} : {}'.format(t[1], label))

    if label == t[2]:
        count += 1
        db_count[t[2]][0] += 1


for k, v in db_count.items():
    print('`{}` rate: {}/{}'.format(k, v[0], v[1]))
print('Global rate: {}'.format(count / len(test_set)))
