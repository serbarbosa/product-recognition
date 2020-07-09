'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil
import numpy as np

from sklearn.cluster import KMeans
from . import img_recognizer
import random



def validate_model(test_imgs_paths, img_feats_hist, numb_of_features, kmeans_model, n_dic):

    random.shuffle(test_imgs_paths)

    hit, miss = 0

    for product_path in test_imgs_paths:
        #get all images filenames
        imgs_paths = os.listdir(product_path)

        for path in imgs_paths:
            # for each product image, tries to recognize it
            # function will return True if product was successfully recognized
            if img_recognizer.recognize(path, img_feats_hist, numb_of_features, kmeans_model, n_dic)
                hit += 1
            else:
                miss += 1







