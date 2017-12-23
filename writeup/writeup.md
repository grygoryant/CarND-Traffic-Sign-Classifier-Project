#**Traffic Sign Recognition** 

##Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: distribution_chart.png "Class distribution"
[image2]: augmented_images.png "Augmented images"
[image3]: 1.jpg "1"
[image4]: 2.jpg "2"
[image5]: 3.jpg "3"
[image6]: 4.jpg "4"
[image7]: 5.jpg "5"

Here is a link to my [project code](https://github.com/grygoryant/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

In order to visualize the dataset I've decided to plot the random image from that dataset and to show the class distribution in the training dataset.

Results of the training dataset analysis can be used later to improve the data augmentation technique. For example, we can check, which classes have insufficient number of samples to augment them.

Here's the training dataset class distribution chart:

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I used simple data preprocessing here, so we just normalize the data. Also, I don't convert image to grayscale since 3 channels carry more information about the given road sign image.

In addition to data normalization I used data augmentation technique which allows us to get more additional data from the given dataset.

To augment testing data I use four operations:

* Rotation
* Translation
* Shear
* Brigthness adjustment

Each of theese operations is applied to image with random parameters from the given range. In order to reduce memory consumption I augment only the current batch.

Here is an example of augmented data:

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As the base model I used the LeNet model from course. After adding some improvements my model performanced approx. 0.96 validation accuracy.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 1x1       | 1x1 stride, valid padding, output 14x14x6     |
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16    |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Convolution 1x1       | 1x1 stride, valid padding, output 5x5x16      |
| Flattening        	| output 400          							|
| Fully connected		| output 120      								|
| RELU					|												|
| Dropout				| keep_prob during training = 0.5				|
| Fully connected		| output 84										|
| RELU  				| 												|
| Dropout				| keep_prob during training = 0.5				|
| Fully connected		| output n_classes								|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with learning rate 0.001. Also, the batch size is 128, but after the data augmentation step it becomes 1280. I trained my model for 50 epochs which gives us 0.96 validation accuracy rather stable.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.998
* validation set accuracy of 0.963 
* test set accuracy of 0.946

I took the LeNet model from course, which was giving 0.83 accuracy. After that I added dropouts between flat layers. Dropouts are used to avoid overfitting by dropping out units of neural network.
Also I've added 1x1 convolutions, which gave me slight increase of accuracy and pushed it closer towards 0.97 (smth like 0.967 in several training epochs).
I've increased the number of epochs to give the model more chances to reach necessary accuracy.

The LeNet model was taken since it's rather simple and accurate model for image classification.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because it's covered by snow.
Other images are rather simple to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of no passing   	| End of all speed and passing limits 			| 
| Yield     			| Yield 										|
| Speed limit 50 km/h	| Speed limit 50 km/h	  						|
| Priority road	   		| Priority road					 				|
| No passing			| No passing      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
It is also necessary to mention that the prediction on the first sign was rather close to the true label, since End of no passing and End of all speed and passing limits signs are rather similar.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Tob 5 predictions for End of no passing:

1. End of all speed and passing limits 4.28012
2. Turn left ahead 3.18339
3. Go straight or right 2.84459
4. End of no passing 2.54888
5. Roundabout mandatory 2.44074

Tob 5 predictions for Yield:

1. Yield 79.5137
2. Priority road 47.314
3. Keep right 46.0459
4. Speed limit (50km/h) 42.4147
5. Road work 32.8599

Tob 5 predictions for Speed limit (50km/h):

1. Speed limit (50km/h) 30.4385
2. Speed limit (30km/h) 25.2344
3. Yield 17.6193
4. Priority road 14.6096
5. Keep right 13.102

Tob 5 predictions for Priority road:

1. Priority road 16.3562
2. Keep right 4.06586
3. Yield 3.66944
4. No passing 3.5279
5. No entry 3.21856

Tob 5 predictions for No passing:

1. No passing 30.2497
2. No entry 14.7287
3. No passing for vehicles over 3.5 metric tons 11.2928
4. Priority road 0.876906
5. Dangerous curve to the right -1.38015

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


