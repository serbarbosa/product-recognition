import os, shutil
import numpy as np
import cv2
from source.preprocess import preprocess_img
from sklearn.cluster import KMeans


def recognize(query_img_path, img_feats_hist, numb_of_features, n_dic):
    '''
        Process and recognizes the product in the image

        query_img_path : path to the image
        numb_of_features : max ammount of features to extract
        n_dic : number of clusters used in kmeans
        img_feats_hist : histograms of features from the trained model
    '''

    # extracting img data to compare
    img_feats = preprocess_and_extract(query_img_path, numb_of_features)
    y = kmeans_model.predict(img_feats[0].astype(np.float32))
    img_hist, _ = np.histogram(y, bins=range(n_dic+1), density=True)

    # computes dist to every possibility -> TODO parallelize
    dists = []
    for img_hist in img_feats_hist:
        dist_ref = []
        # computes distance
        dist_ref.append(np.sqrt(np.sum((img_hist[0]-test_hist)**2)))
        # passes along the path reference of the image
        dist_ref.append(img_hist[1])
        dists.append(dist_ref)

    # find nearest img
    #nearest_path = min(dists, key = lambda x: x[0])
    dists.sort(key = lambda x: x[0])
    nearest_path = dists[:7]

    # plotting query result TODO -> return the class value found instead
    import imageio
    imgq = imageio.imread(test_path)

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

    plt.figure(figsize=(12,8))
    plt.subplot(331); plt.imshow(imgq)
    plt.title('Query'); plt.axis('off')

    imgs = []
    for i in range(len(nearest_path)):
        imgs.append(imageio.imread(nearest_path[i][1]))
        plt.subplot(3,3,i+2)
        plt.imshow(imgs[i])
        plt.title('dist: %.4f' % nearest_path[i][0])
        plt.axis('off')
    plt.show()



