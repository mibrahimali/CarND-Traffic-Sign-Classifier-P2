# Traffic Signs Recognition system using Deep Learning Solution

Nowdays the technology focus on creating  Autonomous vehicles and this field and one of the important aspects associated with it is that teching vehicles how to detect and recognis traffic signs and lights is crucial as teaching it how to drive itself

In this project, i`m purposing a pipeline for traffic signs recognition using Deep learning methods. thes model get use of official german traffic signs dataset for training and testing purposes. 


---

The following sections descripe dataset analysis, model architecture, evaluations and future improvements.

[//]: # (Image References)

[image1]: ./readme_images/dataset_samples.png "Dataset Samples Visualization"
[image2]: ./readme_images/dataset_hist.png "Training Dataset Histogram"
[image3]: ./readme_images/confusion_matrix.png "Confustion matrix of validation dataset"
[image4]: ./readme_images/test_samples.png "Test Samples"
[image5]: ./readme_images/test_samples_reuslts.png "Test Samples Results "
[image6]: ./readme_images/inception_layer_featuremaps.png "Featuremaps visualization of Inception layer 1"
[image7]: ./readme_images/inception_block.png "Inception Block"
[image8]: ./readme_images/Network_model.png "Network Model"


## Data Set Summary & Exploration


I used the Numpy library to calculate summary statistics of the traffic signs data set:

    The size of training set is 34799
    the size of Validation set is 4410
    The size of test set is 12630
    The shape of a traffic sign image is (32, 32, 3)
    The number of unique classes/labels in the data set is 43

the following figure shows samples of training dataset

![alt text][image1]

The code for this step is contained in the first five code cells of the IPython notebook.

Using Pandas Liberary, Histogram of Training data is plotted to give intuation on overall distrpution 
![alt text][image2]

The code for this step is contained in the sixth code cells of the IPython notebook.


### Design and Test a Model Architecture

__Dataset Preprocessing__

Frist i tried image normalization technique using subtracting mean images obtained from training dataset but this step tends to give worse accuracy.
So dataset shuffling is the only pre-processing step for dataset.

__Model Architecture__

my solution is inspired by [1], mainly using spatial transformater networks[2] to nullify any disturtion in traffic sign image due to translation, rotation or even contrast variation.

after that using inception model used in GoogleNet[3]. for feature extraction and classification. the diffrence between purpoced solution here and in [1] is that this purposal tries to minimize model arch. by using fewer layers to optain high accuracy.


__**Localization Network**__
I used LeNet network as my loclization network to learn affine transformation parameters. also add dropout layer to prevent overfitting in training phase.

__**Inception Block**__
the next figure visualize optimized inception block used as key element of my network
![alt text][image7]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (1) Input         		| 32x32x3 RGB image   							| 
| (2) Convolution 1x1     	| 3x3 stride, same padding, outputs 11x11x24 	|
| (3) RELU					|												|
| (4) Convolution 3x3	    | 1x1 stride, input(1)  outputs 11x11x16   		|
| (5) RELU					|												|
| (6) Convolution 5x5		| 1x1 stride, input(1)  outputs 11x11x8   		|
| (7) RELU					|												|
| (8) Max pooling 3x3      	| 1x1 stride, input(4)  outputs 11x11x16 		|
| (9) concatnate(2,4,6,8)	| outputs 11x11x64 feature maps					|


My final model is presented in next figure:
![alt text][image8]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (1)Input         		| 32x32x3 RGB image   							| 
| (2) Spatial Transformer| LeNet Network, outputs 32x32x3 transformed image|
| (3) Inception 3a		| input 32x32x3 RGB image, output 11x11x64 feature maps|
| (4) Dropout			| Keep Probabilit = 0.5							|
| (5) Inception 4a		| input 11x11x64 RGB image, output 4x4x64 feature maps|
| (6) flatten			| output 1024 feature array						|
| (7) Fully connected	| output 512    								|
| (8) RELU					|												|
| (9) Fully connected		| output 100   									|
| (10) RELU					|												|
| (11) Dropout			| Keep Probabilit = 0.5							|
| (12) Fully connected		| output 43 class score        									|
| (13) Softmax				| output 43 class probabilites        									|



The code for training the model is located in  cells  [27-30] of the ipython notebook. 

After tunning, the below parameters were found to yield the best results:
* Learning rate : 0.0001
* Batch size : 128
* Epoch count : 50
* Keep probability for Loclization network : 0.4
* Keep probability for feature maps  : 0.5
* Keep probability for fully connected layer : 0.5

My final model results were:
* training set accuracy of 99.4 %
* validation set accuracy of 95.4%
* test set accuracy of 93.2%


__Preformance measure__

as a preformancce measure for my network, the following figure contain a visualization of confustion matrix calculated using validation dataset
![alt text][image3]


### Test a Model on New Images

Here are ten German traffic signs that I found on the web:

![alt text][image4]

__Notes about test images:__
* Turn left ahead sign is very blury and sky background of sign almost has same color of sign itself
* Keep Right sign has a very high contrast
* Slippery road sign is very distorted 
* gaussian noise is manually add to Speed limit 70 km/h  

The code for making predictions on my final model is located in the 38th cell of the Ipython notebook.
Here are the results of the prediction:
![alt text][image5]

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.2 %

__**Comments on Preformance of small test set**__

* Spatial Transformer Network was successfully focus on the spacific reagion of interset and asly denoising most of images.
* Network missclassified speed limit sign 70km/h as speed limit sign 20km/h with confidance 84.2% and correct class was given probability of 5%. this can be due to add guassian noise 
* Network missclassified speed limit sign 60km/h as speed limit sign 50km/h with confidance 96.7% and correct class was given probability of 2.64%

As a final step, visualization of inner feature maps preduced my frist inception block can be found in next figure.
![alt text][image6]

this visualization can help understand basic features learnt by network to prediect its outputs, aslo help detecting of overfitting if happen 

### Further Improvement

1. Use Data Augmentation technequies for dataset balancing 
2. Incease complixty of inception block by design a separet 1x1 conv. layer be each path of inception block instead of only single 1x1 conv for all
3. Making Network model deeper by using more inception blocks 


### Refrences

1. Mrinal Haloi 2015 "[Traffic Sign Classification Using Deep Inception Based Convolutional Networks](https://arxiv.org/abs/1511.02992)". arXiv:1511.02992
2. Max Jaderberg and Karen Simonyan and Andrew Zisserman and Koray Kavukcuoglu 2015 "[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)". arXiv:1506.02025
3. Christian Szegedy and Vincent Vanhoucke and Sergey Ioffe and Jonathon Shlens and Zbigniew Wojna 2015 "[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)". arXiv:1512.00567
4. https://github.com/daviddao/spatial-transformer-tensorflow