import cv2
import Levenshtein as leven
import os.path
import pickle

class Freeman:
    def __init__(self,database):
        self.database = database
        self.freeman_of = { }

        self.preprocess()

    def preprocess(self,):
        if not os.path.isfile('freeman_dump.pkl'):
            for i in range(len(self.database)):
                self.compute_freeman(i)
            pickle.dump(self.freeman_of, open('freeman_dump.pkl', 'wb'))
        else:
            self.freeman_of = pickle.load(open('freeman_dump.pkl', 'rb'))

    def compute_freeman(self,i):
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

    def rotate(self,strg,n):
        return strg[n:] + strg[:n]

    def dist(self,i1, i2):
        if not i1 in self.freeman_of:
            self.compute_freeman(i1)
        f1 = self.freeman_of[i1]

        if not i2 in self.freeman_of:
            self.compute_freeman(i2)
        f2 = self.freeman_of[i2]

        res = 1000*1000
        return min([leven.distance(f1, self.rotate(f2, i)) for i in range(1,len(f2), 50)])
