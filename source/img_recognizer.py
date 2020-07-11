import os, shutil, sys
import numpy as np
import cv2
from .features_extractor import _preprocess_and_extract
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter


def recognize(query_img_path, img_feats_hist, numb_of_features, kmeans_model, n_dic):
    '''
        Process and recognizes the product in the image

        query_img_path : path to the image
        img_feats_hist : histograms of features from the trained model
        numb_of_features : max ammount of features to extract
        kmeans_model : the kmeans model for the training set
        n_dic : number of clusters used in kmeans

        return : string, string - the product code of the image closest to the query
                                and an evaluation on the result based on a threshold
    '''


    # extracting img data to compare
    query_img_feats = _preprocess_and_extract(query_img_path, numb_of_features)
    if query_img_feats[1] == 0: # no features were found
        return 'ERROR'

    y = kmeans_model.predict(query_img_feats[0].astype(np.float32))
    query_img_hist, _ = np.histogram(y, bins=range(n_dic+1), density=True)

    # computes dist to every possibility
    dists = []
    for img_hist in img_feats_hist:
        dist_ref = []
        # computes distance
        dist_ref.append(np.sqrt(np.sum((img_hist[0]-query_img_hist)**2)))
        # passes along the path reference of the image
        dist_ref.append(img_hist[1])
        dists.append(dist_ref)


    # finds the nearest image and extracst the product code
    prod_guess = min(dists, key = lambda x: x[0])
    prod_code= prod_guess[1].split(os.sep)[-2]
    # evaluates if the distance to the closest image is really close enough
    threshold = 0.065
    if prod_guess[0] > 0.065:
        # the program evalutes that the image is not in the base
        # and that the prediction was wrong
        return prod_code, "out"
    else:
        # the program evalutes that the prediction is correct
        return prod_code, "in"



def recognize_dist(query_img_path, img_feats_hist, numb_of_features, kmeans_model, n_dic):
    ''' Returns the closest image to the query and the distance between them.'''

    # extracting img data to compare
    query_img_feats = _preprocess_and_extract(query_img_path, numb_of_features)
    if query_img_feats[1] == 0: # no features were found
        return 'ERROR'

    y = kmeans_model.predict(query_img_feats[0].astype(np.float32))
    query_img_hist, _ = np.histogram(y, bins=range(n_dic+1), density=True)

    # computes dist to every possibility -> TODO parallelize
    dists = []
    for img_hist in img_feats_hist:
        dist_ref = []
        # computes distance
        dist_ref.append(np.sqrt(np.sum((img_hist[0]-query_img_hist)**2)))
        # passes along the path reference of the image
        dist_ref.append(img_hist[1])
        dists.append(dist_ref)

    # find nearest img
    nearest_path = min(dists, key = lambda x: x[0])

    #get product code
    prod_code = nearest_path[1].split(os.sep)[-2]


    # plots the closest products and their distance
    dists.sort(key = lambda x: x[0])
    nearest_path = dists[:8]

    # plotting query result
    import imageio

    #imgq = imageio.imread(query_img_path)

    #plt.figure(figsize=(12,8))
    #plt.subplot(321); plt.imshow(imgq)
    #plt.title('Query'); plt.axis('off')

    #imgs = []
    #imgs.append(imageio.imread(nearest_path[1]))
    #plt.subplot(3,2,2); plt.imshow(imgs[0])
    #plt.title("closest: %.4f" % nearest_path[0])
    #plt.axis("off")
    #plt.show()

    #plt.figure(figsize=(12,8))
    #plt.subplot(331); plt.imshow(imgq)
    #plt.title('Query'); plt.axis('off')

    plt.figure(figsize=(15,15))
    columns=3
    rows=3
    w,h = 10,10
    imgq = imageio.imread(query_img_path)
    plt.subplot(rows, columns, 1)
    plt.imshow(imgq)
    plt.title('Query'); plt.axis('off')

    imgs = []
    for i in range(1, len(nearest_path) + 1):
        imgs.append(imageio.imread(nearest_path[i-3][1]))
        plt.subplot(rows,columns,i+1)
        plt.imshow(imgs[i-1])
        plt.title('dist: %.4f' % nearest_path[i-1][0])
        plt.axis('off')
    plt.show()

    # returns the closest product and its distance to the query img
    return prod_code, nearest_path[0]


