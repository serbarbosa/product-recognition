'''
     author: Sergio Ricardo Gomes Barbosa Filho
     nusp:   10408386
     course: scc0251
     year/semester: 2020/1
'''
import bovw
import os, sys
from source import kmeans_handler, img_recognizer, validate
import create_bases
import random


scritp_dir = os.getcwd()
test_imgs_path = os.path.join(scritp_dir, 'test')
train_imgs_path = os.path.join(scritp_dir, 'train')

# choosing parameters
numb_of_features = 1000
n_dic = 130

# creating trainig and testing bases
create_bases.create(600, test_entropy=0.15)

# buiding model
bovw.perform_bovw(numb_of_features, n_dic)

# loading model for testing
dictionary, kmeans_model = kmeans_handler.load_model()
validate.validate_model(test_imgs_path, train_imgs_path, dictionary, numb_of_features, kmeans_model, n_dic)

products_in_train = os.listdir(train_imgs_path)

# choosing query images - retriving some randomly from the test base
products_paths = []
products = os.listdir(test_imgs_path)
random.shuffle(products)
for prod in products[:1]:
    # will enter the products folder and choose an image randomly
    imgs = os.listdir(os.path.join(test_imgs_path, prod))
    random.shuffle(imgs)
    products_paths.append(os.path.join(test_imgs_path, prod, imgs[0]))

# make predictions
#for test_path in products_paths:
#    test_prod = test_path.split(os.sep)[-2]
#    if test_prod in products_in_train:
#        print("---------\nesta no treino")
#    else:
#        print("nao esta no treino")
#    pred = img_recognizer.recognize_dist(test_path, dictionary, numb_of_features, kmeans_model, n_dic)
#
#    print("dist = " + str(pred[1][0]))
#    if(pred[0] == test_prod):
#        print("yes")
#    else:
#        print("no")





