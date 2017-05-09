import cv2
import Levenshtein as leven
import os.path
import pickle

class freeman:
    def __init__(self,database, train_set):
        self.database = database
        self.train_set = train_set#useless

        self.freeman_of = {}

        self.preprocess()

    def preprocess(self,):
        if not os.path.isfile("freeman_dump.pkl"):
            for i in range(len(self.database)):
                print("Processing "+str(i)+"/"+str(len(self.database))+"...")
                self.get_freeman(i)
            print("Dumping...")
            pickle.dump(self.freeman_of, open("freeman_dump.pkl", "wb"))
        else:
            print("Loading freeman dump...")
            self.freeman_of = pickle.load(open("freeman_dump.pkl", "rb"))

    def get_freeman(self,i):
        im2, contours, hierarchy = cv2.findContours(self.database[i][0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = [0,0] # 0: length, 1: contour
        for i_c,c in enumerate(contours):
            if c.shape[0] > max_contour[0]:
                max_contour[0] = c.shape[0]
                max_contour[1] = i_c
        self.freeman_of[i] = self.get_freeman_code(contours[max_contour[1]])

    def get_freeman_code(self,cnt):
        dirs = {(0,1):'0',(1,0):'1',(0,-1):'2',(-1,0):'3',(1,1):'4',(-1,-1):'5',(-1,1):'6',(1,-1):'7'}
        free_man = ''
        p = cnt[0][0]
        for pp in cnt[1:]:
            delta = pp[0]-p
            p = pp[0]
            free_man += dirs[(delta[0],delta[1])]
        return free_man


    def dist(self,i1, i2):
        if not i1 in self.freeman_of:
            f1 = self.get_freeman(i1)
        else:
            f1 = self.freeman_of[i1]

        if not i2 in self.freeman_of:
            f2 = self.get_freeman(i2)
        else:
            f2 = self.freeman_of[i2]

        #print(i1,i2)
        return leven.distance(f1,f2)


class freeman_median:
    def __init__(self,database, train_set):
        self.database = database
        self.train_set = train_set#useless

        self.freeman_of = {}

        self.preprocess()

    def preprocess(self,):
        if not os.path.isfile("freeman_dump.pkl"):
            for i in range(len(self.database)):
                print("Processing "+str(i)+"/"+str(len(self.database))+"...")
                self.get_freeman(i)
            print("Dumping...")
            pickle.dump(self.freeman_of, open("freeman_dump.pkl", "wb"))
        else:
            print("Loading freeman dump...")
            self.freeman_of = pickle.load(open("freeman_dump.pkl", "rb"))

        self.equi_class = {}
        for i in range(len(self.database)):
            the_class = self.database[i][2]
            if not the_class in self.equi_class:
                self.equi_class[the_class] = []
            self.equi_class[the_class].append(self.freeman_of[i])
        for the_class in self.equi_class:
            print("Processing median of "+the_class+"...")
            self.equi_class[the_class] = leven.median(self.equi_class[the_class])

    def get_freeman(self,i):
        im2, contours, hierarchy = cv2.findContours(self.database[i][0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = [0,0] # 0: length, 1: contour
        for i_c,c in enumerate(contours):
            if c.shape[0] > max_contour[0]:
                max_contour[0] = c.shape[0]
                max_contour[1] = i_c
        self.freeman_of[i] = self.get_freeman_code(contours[max_contour[1]])

    def get_freeman_code(self,cnt):
        dirs = {(0,1):'0',(1,0):'1',(0,-1):'2',(-1,0):'3',(1,1):'4',(-1,-1):'5',(-1,1):'6',(1,-1):'7'}
        free_man = ''
        p = cnt[0][0]
        for pp in cnt[1:]:
            delta = pp[0]-p
            p = pp[0]
            free_man += dirs[(delta[0],delta[1])]
        return free_man


    def dist(self,i1, i2):
        class1 = self.database[i1][2]
        f1 = self.equi_class[class1]
        f2 = self.freeman_of[i2]
        return leven.distance(f1,f2)


