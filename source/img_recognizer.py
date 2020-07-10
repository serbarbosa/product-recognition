import os, shutil
import numpy as np
import cv2
from .features_extractor import _preprocess_and_extract
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def recognize(query_img_path, img_feats_hist, numb_of_features, kmeans_model, n_dic):
    '''
        Process and recognizes the product in the image

        query_img_path : path to the image
        img_feats_hist : histograms of features from the trained model
        numb_of_features : max ammount of features to extract
        kmeans_model : the kmeans model for the training set
        n_dic : number of clusters used in kmeans

        return : string - 'SUCCESS', 'FAILURE' or 'ERROR' in the recognizing task
    '''


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

    #get query product code
    q_prod_code = query_img_path.split(os.sep)[-2]

    # returns True if the producs are the same
    if prod_code == q_prod_code:
        return 'SUCCESS'
    else:
        return 'FAILURE'

    #dists.sort(key = lambda x: x[0])
    #nearest_path = dists[:7]

    # plotting query result
    #import imageio
    #imgq = imageio.imread(query_img_path)

    #plt.figure(figsize=(12,8))
    #plt.subplot(321); plt.imshow(imgq)
    #plt.title('Query'); plt.axis('off')
    #
    #imgs = []
    #imgs.append(imageio.imread(nearest_path[1]))
    #plt.subplot(3,2,2); plt.imshow(imgs[0])
    #plt.title("closest: %.4f" % nearest_path[0])
    #plt.axis("off")
    #plt.show()

    #plt.figure(figsize=(12,8))
    #plt.subplot(331); plt.imshow(imgq)
    #plt.title('Query'); plt.axis('off')

    #imgs = []
    #for i in range(len(nearest_path)):
    #    imgs.append(imageio.imread(nearest_path[i][1]))
    #    plt.subplot(3,3,i+2)
    #    plt.imshow(imgs[i])
    #    plt.title('dist: %.4f' % nearest_path[i][0])
    #    plt.axis('off')
    #plt.show()



