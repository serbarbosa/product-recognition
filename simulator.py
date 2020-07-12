'''
     author: Sergio Ricardo Gomes Barbosa Filho
     nusp:   10408386
     course: scc0251
     year/semester: 2020/1
'''

import bovw
import numpy as np
import os, sys
from source import kmeans_handler, img_recognizer, validate
import create_bases
import random


scritp_dir = os.getcwd()
train_imgs_path = os.path.join(scritp_dir, 'train')
test_imgs_path = os.path.join(scritp_dir, 'test')
# An extra test, with images different from the above, to validate the simulation
extra_test_path = os.path.join(scritp_dir, 'test_extra')


# First we create a small train base and a test base
# with only unknown products
create_bases.create(1, test_entropy=1, test_size = 2000) # test_size is the numb of test images

# Then we define the parameters for the bag of words
numb_of_features = 3000
n_dic = 150

# Creating the dictionary of visual words and computing the descriptors histogram
# for each image
bovw.perform_bovw(numb_of_features, n_dic)
dictionary, kmeans_model = kmeans_handler.load_model()

# loading test images names and paths from the test base
test_products = os.listdir(test_imgs_path)
test_imgs = []
test_imgs_paths = []
for product in test_products:
    imgs = os.listdir(os.path.join(test_imgs_path, product))
    test_imgs += imgs
    test_imgs_paths += [os.path.join(test_imgs_path, product, img) for img in imgs]

# We'll try to keep track of which classes are learned
# during the simulation
classes_learned = {}

# Now we try to recognize the images
# If the image is said to be unknown, the computed histogram is fed
# to the model
for i in range(len(test_imgs)):
    if os.path.exists(test_imgs_paths[i]):
        prediction = img_recognizer.recognize_or_get_histogram(test_imgs_paths[i], dictionary,
                                                               numb_of_features, kmeans_model, n_dic, threshold=0.04)
        product_name = test_imgs_paths[i].split(os.sep)[-2]
        # Checking if the model has found a valid answer
        if prediction[1] == "out":
            text = "product " + product_name  + " wasnt recognized"
            # Then the image was considered unknown and is added to the dictionary
            # but only if it hasn't been added more than 4 times befor
            if product_name not in classes_learned:
                classes_learned[product_name] = 0
            if classes_learned[product_name] < 5:
                classes_learned[product_name] += 1
                dictionary = np.vstack([dictionary, np.array(prediction[2])])
                text += " and was added to the model."
            print(text)
        else:
            # A result was given and we can test if it is correct
            if prediction[0] == test_imgs_paths[i].split(os.sep)[-2]:
                print("---- product " + product_name + " successfuly learned and identified. ----")
                # product successfully learned. We update the 'classes_learned' dict
                # to be sure it will be taken into account in the validation
                classes_learned[product_name] += 3
                #input("\n-------- press <enter> to continue ---------\n")
            else:
                print("product " +  product_name + " mistakenly identified as " + prediction[0])

    else:
        print("erro")

# Computing how many new classes were successfully learned
classes_added = 0
for k, v in classes_learned.items():
    if v > 2:
        classes_added += 1
print("\n%d novos produtos foram aprendidos." % classes_added)

# At the end, we can validate the resulting dictionary
validate.validate_model(extra_test_path, train_imgs_path, dictionary, numb_of_features, kmeans_model, n_dic, new_classes=classes_learned)



