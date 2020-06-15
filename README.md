# product-recognition
Author: Sérgio Ricardo G. B. Filho  
NUSP: 10408386


### Project Goal
---------
The goal of this project is to recognise supermarket products in images or, more specifically, every product of a supermarket as individual products in images of the checkout conveyor belt. The model created will be able to generate representations for new products(not present in the training base) in order to recognise them. The idea is that after an unknown product appears a labeled representation is stored and when this product appears for the forth time the program will be able to recognise it using only the three previous representations.  
</br> 
### Input Images
---------
The images were provided by the gryfo company.  
Each image contains a close up of a single product in a specic orientation. That way each product can be represented by a set of images showing different faces and rotatios. See the example below for a cherry gelatin.
<p float="left">
  <img src="source/sample_product/1.jpg" width='250'>
  <img src="source/sample_product/3.jpg" width='250'>
  <img src="source/sample_product/5.jpg" width='250'>
  <img src="source/sample_product/6.jpg" width='250'>
</p>
</br> 

### Steps
---------
As a first attempt to achieve the goal is going to be used a computer vision approach for feature descriptions extraction.  
Before extracting the features a preprocessing step is performed to improve the contrast in the image. For this step a clahe filter is applied.
<p float="left">
  <img src="source/partial_examples/original.jpg" width='250'>
  <img src="source/partial_examples/clahe.jpg" width='250'>
</p>
Notice that the use of clahe added more noise to the image so, to avoid extracting noise as features later on(example below), a denoise step is required.
<p float="left">
  <img src="source/partial_examples/no_denoise.jpg" width='250'>
  <img src="source/partial_examples/denoised.jpg" width='250'>
</p>
To perform the feature extraction step the ORB algorithm was chosen as it has proven to be as efficient as SIFT but faster and free for commercial use. Below are the results for the cherry gelatin example. In the image on the left without preprocessing 796 features were extracted whereas in the image on the right, after the preprocessing steps, 1119 features were extracted.	
<p float="left">
  <img src="source/partial_examples/out_orb.jpg" width='250'>
  <img src="source/partial_examples/out_prep_orb.jpg" width='250'>
</p>  
For the next steps it will be necessary to decide which ML or computer vision method to use to retrive the product name for a given input image.
It has to be taken into account that the different rotations and scales in the images can be hard to deal with so changes may be needed in the approach used so far.
</br>

### First Results
---------
A Brute Force method was used to compare the different images of the product and match the 30 most similar features. Notice that the greater the transformation of the image, the greater the number of wrong matches.
<p float="left">
  <img src="source/partial_examples/1_and_2.jpg" width='250'>
  <img src="source/partial_examples/3_and_4.jpg" width='250'>
  <img src="source/partial_examples/5_and_6.jpg" width='250'>
  <img src="source/partial_examples/3_and_7.jpg" width='250'>
</p>
