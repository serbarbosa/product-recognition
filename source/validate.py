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

def validate_model(test_imgs_path, train_imgs_path, img_feats_hist, numb_of_features, kmeans_model, n_dic, new_classes=None):
    tests_paths = os.listdir(test_imgs_path)


    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    imgs_paths = []
    true_classes = []       # to store correct answers

    # mapping classes in the train base to check if product is unknown
    classes_in_train = {}
    for prod_class in os.listdir(train_imgs_path):
        classes_in_train[prod_class] = 0
    classes_in_train = os.listdir(train_imgs_path)

    # if 'new_classes' is not None, we need to add those classes to the known ones
    if new_classes is not None:
        for prod, occurrences in new_classes.items():
            if occurrences > 2: # then the class is supposed to be learned
                classes_in_train.append(prod)


    for product_path in tests_paths:

        product_imgs = os.listdir(os.path.join(test_imgs_path, product_path))
        random.shuffle(product_imgs)
        # chooses up to 3 images for each product
        for img_path in product_imgs[:3]:

            imgs_paths.append(os.path.join(test_imgs_path, product_path, img_path))
            if product_path in classes_in_train:
                # the product should be recognized
                true_classes.append(product_path)
            else:
                # product cant be recognized because isnt in the training base
                true_classes.append("out")
    # processing everything in parallel
    predictions = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(img_recognizer.recognize)(
                                os.path.join(img_path, product_path, img_path),
                                img_feats_hist,
                                numb_of_features,
                                kmeans_model,
                                n_dic)
                                for img_path in imgs_paths
                                )
    for i in range(len(true_classes)):

        # unknown product
        if true_classes[i] == "out":
            if predictions[i][1] == "out":
                true_neg += 1
            else:
                false_pos += 1

        # known product
        elif true_classes[i] == predictions[i][0]:
            true_pos += 1
            # can still be a false positive or a false negative
        elif predictions[i][1] == "out":
            false_neg += 1
        else:
            false_pos += 1

    preds_results = []
    for val in predictions:
        if val[1] == "out":
            preds_results.append(val[1])
        else:
            preds_results.append(val[0])

    precision, recall, f1 = 0,0,0
    if(true_pos + false_pos) > 0:
        precision = true_pos/(true_pos + false_pos) # in this real world problem, has to be one
    if(true_pos + false_neg) > 0:
        recall = true_pos/(true_pos + false_neg)    # for this applicatian, measures its efficiency
    if(precision + recall) > 0:
        f1 = 2*(precision*recall)/(precision + recall)

    print("true_positives: %d    true_negatives: %d     false_pos: %d    false_neg: %d" % (true_pos, true_neg, false_pos, false_neg))
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("F1 Score: %f" % f1)
    _evaluate_results(true_classes, preds_results)


def validate_model_v2(test_imgs_path, train_imgs_path, numb_of_features):

    #loading trained data
    clf, stdScaler, n_dic, voc = joblib.load("bovw.pkl")

    tests_paths = os.listdir(test_imgs_path)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0


    imgs_paths = []
    imgs_classes = []

    # mapping classes in the train base to check if product is unknown
    classes_in_train = {}
    for prod_class in os.listdir(train_imgs_path):
        classes_in_train[prod_class] = 0
    classes_in_train = os.listdir(train_imgs_path)

    for product_path in tests_paths:

        product_imgs = os.listdir(os.path.join(test_imgs_path, product_path))
        random.shuffle(product_imgs)
        # chooses up to 3 images for each product
        for img_path in product_imgs[:3]:

            imgs_paths.append(os.path.join(test_imgs_path, product_path, img_path))
            if product_path in classes_in_train:
                # the product should be recognized
                imgs_classes.append(product_path)
            else:
                # product cant be recognized because isnt in the training base
                imgs_classes.append("out")

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


    #true_classes = [desc[2].split(os.sep)[-2] for desc in descriptors_list]
    #predictions = [prediction for prediction in clf.predict(test_features)]

    #print ("true_class ="  + str(true_classes))
    #print ("prediction ="  + str(predictions))
    #_evaluate_results(true_classes, predictions)


def _evaluate_results(true_classes, predictions):

    accuracy = accuracy_score(true_classes, predictions)
    print ("accuracy: %f" % accuracy)
    #cm = confusion_matrix(true_classes, predictions)

    #pl.matshow(cm)
    #pl.title('Confusion matrix')
    #pl.colorbar()
    #pl.show()



