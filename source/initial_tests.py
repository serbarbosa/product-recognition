'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys

from scipy import ndimage
from preprocess import preprocess_img

# At this first attempt we'll extract features from pictures of a product in different angules. 
# Then we'll compare using brute force to see if the features remain the same after transformations
# in the image

#The comparisond will be between images 1 & 2, 3 & 4, 5 & 6 and 3 & 7


def match_and_write(matcher, img1, kp1, desc1, img2, kp2, desc2, file_name):
    ''' Will match descriptors of img1 and img2 using the matcher provided and save the result '''
    
    #matching and sorting to separate better matches
    matches = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)
    d = []
    for m in matches[:30]:
        d.append(m.distance)
    print(d)
    drawn_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)
    cv2.imwrite(file_name, drawn_matches)

#reading and preprocessing images
img1 = preprocess_img(cv2.imread("sample_product/1.jpg"))
img2 = preprocess_img(cv2.imread("sample_product/2.jpg"))
img3 = preprocess_img(cv2.imread("sample_product/3.jpg"))
img4 = preprocess_img(cv2.imread("sample_product/4.jpg"))
img5 = preprocess_img(cv2.imread("sample_product/5.jpg"))
img6 = preprocess_img(cv2.imread("sample_product/6.jpg"))
img7 = preprocess_img(cv2.imread("sample_product/7.jpg"))


# using ORB for feature extraction
orb = cv2.ORB_create()

# extracting keypoints and descriptors for each image
kp1, desc1 = orb.detectAndCompute(img1, None)
kp2, desc2 = orb.detectAndCompute(img2, None)
kp3, desc3 = orb.detectAndCompute(img3, None)
kp4, desc4 = orb.detectAndCompute(img4, None)
kp5, desc5 = orb.detectAndCompute(img5, None)
kp6, desc6 = orb.detectAndCompute(img6, None)
kp7, desc7 = orb.detectAndCompute(img7, None)


# Initializing brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

match_and_write(bf, img1, kp1, desc1, img2, kp2, desc2, "partial_examples/1_and_2.jpg")
match_and_write(bf, img3, kp3, desc3, img4, kp4, desc4, "partial_examples/3_and_4.jpg")
match_and_write(bf, img5, kp5, desc5, img6, kp6, desc6, "partial_examples/5_and_6.jpg")
match_and_write(bf, img3, kp3, desc3, img7, kp7, desc7, "partial_examples/3_and_7.jpg")


