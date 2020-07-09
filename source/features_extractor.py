'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''

import os, shutil
import numpy as np
import cv2
from .preprocess import preprocess_img
from joblib import Parallel, delayed
import pickle

'''
This module will be used to preprocess and extract features from a base os images.
It'll also allow to save and load the features extracted.
'''

def _preprocess_and_extract(img_path, numb_of_features):

    orb = cv2.ORB_create(numb_of_features)
    prep_img = preprocess_img(cv2.imread(img_path))
    key_point, descriptor = orb.detectAndCompute(prep_img, None)

    # no relevant points found
    if descriptor is None:
        descriptor = []
        descriptor = np.array(descriptor)

    return (descriptor, len(descriptor), img_path)

def extract_features(base_path, numb_of_features, save_load=True, overwrite=True):
    '''
        base_path       : path to the directory with the images to extract
        numb_of_features: max amount of features to extract from each image
        save_load       : will save or load(if exits) the extracted features
        overwrite       : will extract and overwrite preexisting files with data
    '''

    descriptors_list = []
    run = True
    if save_load and not overwrite:
        # will see if there is a file with the features to load from
        if os.path.exists("features.pkl"):
            descriptors_list = load_features()
            run = False

    if overwrite or run:

        # listing products images paths
        products_paths = os.listdir(base_path)
        numb_of_features = 1000 # bom com 200

        for product_path in products_paths:

            # preprocessing and extracting images features from training base
            images_paths = os.listdir(os.path.join(base_path, product_path))
            prod_descs = Parallel(n_jobs=-1)(delayed(_preprocess_and_extract)(
                                            os.path.join(base_path,product_path,img_path),
                                            numb_of_features)
                                            for img_path in images_paths
                                            )
            valid_descs = []
            for desc in prod_descs:
                if desc[1] != 0:
                    valid_descs.append(desc)
            descriptors_list += valid_descs

        if save_load:
            # will save the extracted data
            save_features(descriptors_list)

    return descriptors_list

def save_features(feat):

    pickle.dump(feat, open("features.pkl", "wb"))

def load_features():

    feat = pickle.load(open("features.pkl", "rb"))
    return feat







