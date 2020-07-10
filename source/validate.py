'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.vq import vq
from . import img_recognizer
from . import features_extractor
import random
from joblib import Parallel, delayed
import pylab as pl
from sklearn.externals import joblib

def validate_model(test_imgs_path, img_feats_hist, numb_of_features, kmeans_model, n_dic):
    tests_paths = os.listdir(test_imgs_path)

    hit, miss = 0, 0

    imgs_paths = []
    for product_path in tests_paths:

        product_imgs = os.listdir(os.path.join(test_imgs_path, product_path))
        random.shuffle(product_imgs)
        # chooses up to 3 images for each product
        for img_path in product_imgs[:3]:

            imgs_paths.append(os.path.join(test_imgs_path, product_path, img_path))

    # processing everything in parallel
    results = Parallel(n_jobs=-1)(delayed(img_recognizer.recognize)(
                                os.path.join(img_path, product_path, img_path),
                                img_feats_hist,
                                numb_of_features,
                                kmeans_model,
                                n_dic)
                                for img_path in imgs_paths
                                )

    for res in results:
        if res == "SUCCESS":
            hit += 1
        elif res == "FAILURE":
            miss += 1

    print("correctly recognized: " + str(hit) )
    print("not recognized: " + str(miss) )
    print("correctness: " + str(hit*100/(hit+miss)) + "%")


def validate_model_v2(test_imgs_path, numb_of_features):

    clf, stdScaler, n_dic, voc = joblib.load("bovw.pkl")

    tests_paths = os.listdir(test_imgs_path)
    imgs_paths = []
    imgs_classes = []

    for product_path in tests_paths:

        product_imgs = os.listdir(os.path.join(test_imgs_path, product_path))
        random.shuffle(product_imgs)
        # chooses up to 3 images for each product
        for img_path in product_imgs[:3]:

            imgs_paths.append(os.path.join(test_imgs_path, product_path, img_path))
            imgs_classes.append(product_path)   #saves the answer

    # processing everything in parallel
    processed_descriptors = Parallel(n_jobs=-1)(delayed(features_extractor._preprocess_and_extract)(
                                img_path,
                                numb_of_features)
                                for img_path in imgs_paths
                                )
    descriptors_list = []
    for desc in processed_descriptors:
        if desc[1] != 0:    #removes empty ones
            descriptors_list.append(desc)

    #TODO -> realmente necessario fazer essa parte?
    descriptors = np.squeeze(np.array(descriptors_list)[:,[0]])
    bag = np.concatenate(tuple(descriptors[i] for i in range(len(descriptors))), axis=0).astype(np.float32)

    #calculating histogram for the test images features
    test_features = np.zeros((len(descriptors_list), n_dic), "float32")
    for i in range(len(descriptors_list)):
        words, distance = vq(descriptors_list[i][0], voc)
        for w in words:
            test_features[i][w] += 1

    # aplying tf-idf vectorization
    occurrences = np.sum( (test_features > 0) * 1, axis = 0 )
    idf = np.array(np.log((1.0*len(descriptors_list)+1) / (1.0 * occurrences + 1)), "float32")

    # normalizing features
    test_features = stdScaler.transform(test_features)


    true_classes = [desc[2].split(os.sep)[-2] for desc in descriptors_list]
    predictions = [prediction for prediction in clf.predict(test_features)]

    #print ("true_class ="  + str(true_classes))
    #print ("prediction ="  + str(predictions))


    accuracy = accuracy_score(true_classes, predictions)
    print ("accuracy = ", accuracy)
    cm = confusion_matrix(true_classes, predictions)
    print (cm)

    showconfusionmatrix(cm)

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()



