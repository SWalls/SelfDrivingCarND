# Vehicle Detection and Tracking
## By Soeren Walls

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: output_images/vehicle-721.png "Vehicle"
[image1]: output_images/nonvehicle-166.png "Non-Vehicle"
[image2]: output_images/hog-class0-img721-HSV-orient9-ppc8-cpb2.png "HOG example (HSV)"
[image3]: output_images/hog-class0-img842-YCrCb-orient8-ppc8-cpb2.png "HOG example (YCrCb)"
[image4]: output_images/test1-bboxes.png "Bounding Boxes"
[image5]: output_images/test1-heatmap.png "Heatmap"
[image6]: output_images/test1-out.png "Output"
[image7]: output_images/video_frames/frame-20.png "Frame 20"
[image8]: output_images/video_frames/frame-21.png "Frame 21"
[image9]: output_images/video_frames/frame-22.png "Frame 22"
[image10]: output_images/video_frames/frame-23.png "Frame 23"
[image11]: output_images/video_frames/frame-24.png "Frame 24"
[image12]: output_images/video_frames/frame-25.png "Frame 25"
[image13]: output_images/video_frames/frame-26.png "Frame 26"
[image14]: output_images/video_frames/frame-27.png "Frame 27"
[image15]: output_images/video_frames/frame-28.png "Frame 28"
[image16]: output_images/video_frames/frame-29.png "Frame 29"
[image17]: output_images/video_frames/labels-29.png "Frame 29 Labels"
[image18]: output_images/video_frames/bboxes-29.png "Frame 29 BBoxes"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

1\. **Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.**

You're reading it!

### Histogram of Oriented Gradients (HOG)

1\. **Explain how (and identify where in your code) you extracted HOG features from the training images.**

The code for this step is contained in lines 73 through 111 of the file `detect_vehicles.py`. This file contains all of my code for this project.

I started by reading in all the `vehicle` and `non-vehicle` images (lines 116-123). There are roughly 4,000 images in each class. It's important that both classes have approximately the same number of training data points, because this helps avoid unwanted class bias during training. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image0]
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on an image from the `vehicle` data set:

![alt text][image2]

2\. **Explain how you settled on your final choice of HOG parameters.**

I tried various combinations of parameters and decided on the following:

```
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

Here is an example HOG output using these parameters on an image from the `vehicle` data set:

![alt text][image3]

As you can see, whereas the image produced using the `HSV` color space in the previous step clearly describes the outline of a car, this image's outline is not as clear. Despite the fact that the `HSV` color space seems to produce a more intuitive output for HOG as compared to others, the `YCrCb` color space performs consistently better during training.

3\. **Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).**

I trained a linear SVM (at lines 114-183 in my code) using features from HOG, spatial binning, and color histograms, with the following parameters, hard-coded at lines 410-417:

```
color_space = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 16
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
```

The features from all 3 of these operations are combined into a single feature vector of length 8412, which is labeled with either a 1 or a 0, depending on whether it belongs to the `vehicle` or `non-vehicle` class, respectively. Using the above parameters, the accuracy of the SVM on the validation data is 99.1%.

### Sliding Window Search

1\. **Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?**

Instead of using the sliding window search function described in the earlier lessons, which computes the HOG features over and over for each window, I decided to use the `find_cars` function, defined at lines 199-268, which computes the HOG features beforehand, and then performs the sliding window search while re-using the same feature data. This proved much more efficient and saved a lot of time.

In the main `pipeline` function defined at lines 364-408, `find_cars` is called twice at two different scales: 1.5 and 2.5. However, different search boundaries are defined for each scale. Specifically, the algorithm searches with windows at 2.5 scale only in the right-most 300 pixels of the road, since this is where cars are expected to appear large. It searches with windows at 1.0 scale across the entire road, which spans the whole width of the image and 256 pixels high. This can be seen on line 371:

```
for scale, ystart, ystop, xstart, xstop in [[1.5, 400, 656, 0, img_width], [2.5, 380, 656, img_width-300, img_width]]:
```

These scales were chosen because a scale that is too small (such as 0.5) results in a lot of false-positives, and for the most part, scales bigger than 1.5 are redundant, since they seem to identify cars equally as well as the 1.5 scale. As such, these extra scales aren't worth the performance loss to compute.

Instead of overlap, the `find_cars` function defines at line 224 how many cells to jump per step of the sliding window algorithm. I chose to keep the default value of 2 cells per step, since this provides some overlap without being too performance heavy.

2\. **Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?**

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To optimize performance, I combined the hog feature extraction and sliding window search into one function.

First, the pipeline calls the `find_cars` function, which performs the HOG feature extraction and the sliding window search, testing images on the trained classifier in order to get all the bounding boxes of suspected vehicles in the image at 2 different scales. This happens at line 372. Here is an example of the bounding boxes produced by this function:

![alt text][image4]

Next, the pipeline uses the bounding boxes to create a new `Frame` object with a `heatmap`. This can be seen at line 375. In the `Frame` class, at lines 338-349, a flattened version of the heatmap called `flatmap` is created. This is similar to the output of the `label` function, but it does not differentiate between groups of pixels. Instead, it simply takes all the heatmap values greater than 0 and flattens them all down to 1. The reason for doing this is so that heat generated by overlapping bounding boxes on the same frame is a separate metric from the heat generated by overlapping bounding boxes across multiple frames.

This flatmap is computed for each frame, and then all the flatmaps from the previous 20 frames are combined into one single heatmap called `timemap`, at lines 376-380. When this sum is computed, the current frame is weighted as 30% of the sum, while the previous frames count for 70%. This is so that the bounding boxes in the current frame will create more heat in the `timemap` compared to the previous frames. Next, the `timemap` is thresholded at line 384, so that false positives that only appear in a few frames are removed. Here is an example of a thresholded `timemap`, created from the bounding boxes in the previous image:

![alt text][image5]

Finally, the pipeline uses the `labels` function to identify separate cars in the image. These labels are then drawn over the cars to produce the final output:

![alt text][image6]

---

### Video Implementation

1. **Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)**

Here's a [link to my video result](project_solution.mp4)

2. **Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.**

I record the positions of positive detections for 20 frames at a time, and store them in a queue (list) of Frame object called `frames` (at line 361). At each frame in the video, a new `Frame` object is added to the queue which stores the `heatmap` and `flatmap` values described in Step 2 of the previous section. Then, I calculate the sum of all 10 flatmaps in the queue, to get one single heatmap representing the past 10 frames, called `timemap`. This sum is weighted, as described earlier.

Next, I threshold the `timemap` by 8 to remove false positives and identify vehicle positions. This is based on the intuition that a detection is a false positive if its bounding box is overlapped by other bounding boxes in less than ~half of the previous frames. I then use `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assume each blob corresponds to a separate vehicle. Finally, I construct bounding boxes to cover the area of each blob that has been detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are 10 frames and their corresponding heatmaps:
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 10 frames:
![alt text][image17]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image18]

---

### Discussion

1\. **Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The approach I took with this project was to prioritize creating and training a good classifier by playing around with the parameters for HOG and the other image descriptors. After discovering the ideal parameters, I decided to optimize the pipeline by combining the sliding window search and the hog feature extraction into one function, and also keeping a history of detections for 20 frames. This seemed to work well.

One problem I faced was detecing the white car. Many times that I tried to process the video input, it did not detect the white car at all. After browsing through the forums and discovering that other people had this issue as well, I carefully altered my model and made sure that my color spaces were being converted correctly, and that I was correctly scaling the input features for the classifier. Once I had all of this worked out, it began detecting the white car. It still doesn't do it perfectly, but it's much better than before.

In my first submission of this project, my solution was not robust enough, so I heeded the advice of my reviewer, and made a few changes that drastically improved the performance of my algorithm. First of all, I stopped using the `HSV` color-space, and began using the `YCrCb` color space instead. This alone improved the test accuracy of my classifier from 98% to 99.1%. Also, instead of simply accepting the prediction provided by the classifier, I am now using the `svc.decision_function` with a higher threshold to eliminate more false positives. This removed most of the false positives in my video. Finally, I improved the robustness of my pipeline by separating the heat created from single frames and the heat created from multiple frames when thresholding. This made my output much more consistent and predictable. It also made my code more readable, since each frame is now represented by an instance of the `Frame` class.

Despite these improvements, my pipeline will still likely fail on videos where the cars appear in different places in the video, or different sizes, since my pipeline makes certain assumptions about where the cars are and how big they will be.

To make my pipeline even more robust, if I were to continue working on this project, I would spend more time manually separating out the time-sequenced images in the training data so that it doesn't overfit. I would also try to figure out new ways to optimize the pipeline so that the bounding boxes appear a little smoother and show fewer false positives.