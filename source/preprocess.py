'''
    author: Sergio Ricardo Gomes Barbosa Filho
    nusp:   10408386
    course: scc0251
    year/semester: 2020/1
'''
import cv2
import numpy as np


def apply_clahe(input_img):
    '''
        img: rgb image
    '''
    lab = cv2.cvtColor(input_img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    #applied to light
    lab[..., 0] =clahe.apply(lab[..., 0])
    
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

def denoise(input_img):

    return cv2.fastNlMeansDenoisingColored(input_img, templateWindowSize=7, searchWindowSize=21, h=6, hColor=10)

def preprocess_img(input_img):

    #applying clahe for image enhancement
    img = apply_clahe(input_img)
    
    #clahe will add noise to the image
    #we can remove some of the noise
    img = denoise(input_img) 
    
    return img


