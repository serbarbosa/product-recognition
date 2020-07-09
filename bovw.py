'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil
import numpy as np
import cv2
from source.preprocess import preprocess_img
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

import matplotlib.pyplot as plt


def preprocess_and_extract(img_path, numb_of_features):

    orb = cv2.ORB_create(numb_of_features)
    prep_img = preprocess_img(cv2.imread(img_path))
    key_point, descriptor = orb.detectAndCompute(prep_img, None)

    # no relevant points found
    if descriptor is None:
        descriptor = []
        descriptor = np.array(descriptor)

    return (descriptor, len(descriptor), img_path)


train_imgs_path = os.path.join(os.getcwd(), "train")
test_imgs_path = os.path.join(os.getcwd(), "test")
descriptors_list = []
# listing products images paths
products_paths = os.listdir(train_imgs_path)
numb_of_features = 1000 # bom com 200

print("processing training images")
for product_path in products_paths:

    # preprocessing and extracting images features from training base
    images_paths = os.listdir(os.path.join(train_imgs_path, product_path))
    prod_descs = Parallel(n_jobs=-1)(delayed(preprocess_and_extract)(
                                    os.path.join(train_imgs_path,product_path,img_path),
                                    numb_of_features)
                                    for img_path in images_paths
                                    )
    valid_descs = []
    for desc in prod_descs:
        if desc[1] != 0:
            valid_descs.append(desc)
    descriptors_list += valid_descs

print("creating kmeans model")

# set up bag of features
descriptors = np.squeeze(np.array(descriptors_list)[:,[0]])

bag = np.concatenate(tuple(descriptors[i] for i in range(len(descriptors))), axis=0).astype(np.float32)

n_dic = 1000    #-> aumentar dic ->bom com 200
random_state = 1

#defining a KMeans clustering model
kmeans_model = KMeans(n_clusters=n_dic,
                        verbose=False,
                      init='random',
                      random_state=random_state,
                      n_init=3
                      )

#fit the model
kmeans_model.fit(bag)

print("creating descriptors histograms")

#plt.scatter(bag[:, 0], bag[:, 1], c=kmeans_model.labels_)
#plt.axis("off")
#plt.show()

#computing frequency of features in each image
img_feats_hist = []

for entry in descriptors_list:
    y = kmeans_model.predict(entry[0].astype(np.float32))

    hist_bovw, _ = np.histogram(y, bins=range(n_dic+1), density=True)
    img_feats_hist.append([hist_bovw, entry[2]])    # put together the img filename

img_feats_hist = np.array(img_feats_hist)

#using features for recognition  -------  PREDICTION

print("recognizing image")
# get 1 image for first test
test_path = os.path.join(test_imgs_path, "7891048050729", "7891048050729_3452_0.jpg")
test_feats = preprocess_and_extract(test_path, numb_of_features)
y = kmeans_model.predict(test_feats[0].astype(np.float32))
test_hist, _ = np.histogram(y, bins=range(n_dic+1), density=True)

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


