'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from source import *


#defining path for train and test sets
train_imgs_path = os.path.join(os.getcwd(), "train")
test_imgs_path = os.path.join(os.getcwd(), "test")

numb_of_features = 500
n_dic = 300
random_state = 1

print("processing training images")
# obtaining features -> format is (features, amnt_of_features_extracted, image_path)
features_list = features_extractor.extract_features(train_imgs_path, numb_of_features, save_load=True, overwrite=False)

print("creating dictionary of visual words")
# creating visual words dictionary and extracting histograms for each image
img_feats_hist, kmeans_model = kmeans_handler.create_model(features_list, n_dic, random_state, save_load=True, overwrite=False)

print("recognizing image")
# recognizing image from test set
tests_paths = []
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_3452_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_249_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_2037_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_2360_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_3320_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_5836_0.jpg"))
tests_paths.append(os.path.join(test_imgs_path, "7891048050729", "7891048050729_7106_0.jpg"))

for test_path in tests_paths:
    img_recognizer.recognize(test_path, img_feats_hist, numb_of_features, kmeans_model, n_dic)


