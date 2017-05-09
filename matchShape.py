import cv2
import Levenshtein as leven
import os.path
import pickle

class matchShape:
    def __init__(self,database, train_set):
        self.database = database
        self.train_set = train_set#useless

        self.preprocess()

    def preprocess(self,):
        return

    def dist(self,i1, i2):
        return 1-cv2.matchShapes(self.database[i1][0], self.database[i2][0], 3, 0.0) 