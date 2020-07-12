'''
  author: Sergio Ricardo Gomes Barbosa Filho
  nusp:   10408386
  course: scc0251
  year/semester: 2020/1
'''

import numpy as np
import matplotlib.pyploy as plt
from source import img_recognizer, kmeans_handler
import random

scritp_dir = os.getcwd()
train_imgs_path = os.path.join(scritp_dir, 'train')
test_imgs_path = os.path.join(scritp_dir, 'test_extra')

# Loading models
dictionary, kmeans_model = kmeans_handler.load_model()
n_dic, numb_of_features = 150, 3000

products_in_train = os.listdir(train_imgs_path)
# choosing query images - retriving some randomly from the test base
products_paths = []
products = os.listdir(test_imgs_path)
random.shuffle(products)

for prod in products[:15]:  # 15 samples
# will enter the products folder and choose an image randomly
    imgs = os.listdir(os.path.join(test_imgs_path, prod))
    random.shuffle(imgs)
    products_paths.append(os.path.join(test_imgs_path, prod, imgs[0]))

# will predict and output the result and closest images




