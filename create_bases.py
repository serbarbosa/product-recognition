'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import os, shutil, sys
import random
import numpy as np


def create(train_max_size, imgs_per_product=5, test_entropy=0.3, test_size=1.0):
    '''
        Creates train and test bases.

        train_max_size  : int   - Max number of products that can be selected
        imgs_per_product: int   - Number of images to select for trainin for each product.
                                  Products with less images than that are discarded.
        test_entropy : float - Defines the percentual of unknows products in the test base
        test_size       : float - Multiplier to increase the size of test base
    '''
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

    known_products_dirs = products_dirs[:train_max_size]
    unknown_products_dirs = products_dirs[train_max_size+1:]
    random.shuffle(unknown_products_dirs)
    test_count = 0
    # iterating through image directories
    for product_dir in known_products_dirs:
        # retrieving all product images
        images = os.listdir(os.path.join(base_folder, product_dir))
        random.shuffle(images)

        # will get only products with more than the indicated ammount of images
        # the test base must have at least 1 image for each product
        if len(images) > imgs_per_product+1:
            train = []
            test = []

            # copying train images
            train += images[:imgs_per_product]
            os.mkdir(os.path.join(train_dir, product_dir))
            for img in train:
                shutil.copy(os.path.join(base_folder, product_dir, img), os.path.join(train_dir, product_dir, img))

            # copying test images
            # 1 - test_entropy is the chance to get a test sample for this product
            if random.random() < 1.0-test_entropy*test_size:
                #up to 3 test images for each product
                test += images[imgs_per_product:imgs_per_product + 3]
                test_count += len(test)

                os.mkdir(os.path.join(test_dir, product_dir))
                for img in test:
                    shutil.copy(os.path.join(base_folder, product_dir, img), os.path.join(test_dir, product_dir, img))
    # now we need to select the unkown products for the test base
    # up to now only 1-test_entropy of the test base was selected

    # amount to create a base with only unknown products
    unkown_imgs_amnt = train_max_size
    if (test_entropy < 1.0):
        unkown_imgs_amnt = int( test_count*test_entropy/(1.0-test_entropy) )

    count_unknown = 0
    i = 0
    while(count_unknown < unkown_imgs_amnt and i < len(unknown_products_dirs)):
        unk_product = unknown_products_dirs[i]
        i+=1
        imgs = os.listdir(os.path.join(base_folder, unk_product))[:3]
        count_unknown += len(imgs)
        os.mkdir(os.path.join(test_dir, unk_product))
        for img in imgs:
            shutil.copy(os.path.join(base_folder, unk_product, img), os.path.join(test_dir, unk_product, img))


