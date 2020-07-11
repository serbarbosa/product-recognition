'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil, sys
import random
import numpy as np

script_dir = os.getcwd()
# creating train and test directories
train_dir = os.path.join(script_dir, 'train')
test_dir = os.path.join(script_dir, 'test')

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)

os.mkdir(train_dir)
os.mkdir(test_dir)

# retrieving image folders
base_folder = os.path.join(script_dir, "base")

products_dirs = os.listdir(base_folder)
random.shuffle(products_dirs)
# iterating through image directories
for product_dir in products_dirs[:int(sys.argv[1])]:
    # retrieving all product images
    images = os.listdir(os.path.join(base_folder, product_dir))
    random.shuffle(images)
    # products with less than 3 images are not considered
    if len(images) > 8: # ---> troca aqui
        # 3 to 5 images chosen for training base
        # the others will be used for testing
        train = []
        test = []
        train += images[:7]
        test += images[7:]
        #if len(images) > 5: # 5 images
        #    train += images[:5]
        #    test += images[5:]
        #elif len(images) == 6: # 4 images
        #    train += images[:4]
        #    test += images[4:]
        #else:
        #    train += images[:3]
        #    test += images[3:]
        #copying images to their respective folders
        #train
        os.mkdir(os.path.join(train_dir, product_dir))
        for img in train:
            shutil.copy(os.path.join(base_folder, product_dir, img), os.path.join(train_dir, product_dir, img))
        #test
        os.mkdir(os.path.join(test_dir, product_dir))
        for img in test:
            shutil.copy(os.path.join(base_folder, product_dir, img), os.path.join(test_dir, product_dir, img))
