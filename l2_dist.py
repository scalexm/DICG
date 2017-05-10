import cv2

class L2_dist:
    def __init__(self,database):
        self.database = database
        return
    def dist(self,i_query, i_train):
        img1,img2 = self.database[i_query][0],self.database[i_train][0]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)

        r1 = img1[(h1 - h) // 2 : (h1 + h) // 2, (w1 - w) // 2 : (w1 + w) // 2]
        r2 = img2[(h2 - h) // 2 : (h2 + h) // 2, (w2 - w) // 2 : (w2 + w) // 2]
        return cv2.norm(r1, r2)
