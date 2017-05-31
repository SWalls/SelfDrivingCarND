# Advanced Lane Lines
## by Soeren Walls

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: output_images/undistort_chessboard.jpg "Undistorted Chessboard"
[image1]: output_images/straight_lines1.jpg "Unaltered Input"
[image2]: output_images/straight_lines1_undist.jpg "Road Transformed"
[image3]: output_images/straight_lines1_combo_masked.jpg "Binary Example"
[image4]: output_images/straight_lines1_warped.jpg "Warp Example"
[image5]: output_images/straight_lines1_polylanes_vis.jpg "Fit Visual"
[image6]: output_images/straight_lines1_output.jpg "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

1\. **Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.** 

You're reading it!

### Camera Calibration

1\. **Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.**

The code for this step is contained in the `calibrate_camera` function, in lines 91-128 of the file `lanelines.py`. This file contains all the code used for this project.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image0]

### Pipeline (single images)

1\. **Provide an example of a distortion-corrected image.**

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image1]
![alt text][image2]

I begin by collecting the camera matrix and the distortion coefficients from the `calibrate_camera` function described in the previous step (code line 540). Later, in the pipeline on line 577, I pass these as parameters in a call to the `undistort` function described in lines 130-134, which uses the `cv2.undistort()` function. Since the camera matrix and distortion coefficients are stored in global variables, I can use them at any time to undistort an image.

2\. **Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.**

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 218-238). It combines the gradients in both the X and Y direction, the magnitude of both gradients, the direction of both gradients, and the S channel of the HLS colorspace (thresholded).

However, instead of simply taking the direction of the gradients, I use the `cv2.erode` and `cv2.dilate` functions on the resulting binary image of directions (lines 226-229), so that only relatively big clusters of white pixels remain. By removing all the isolated white pixels in the `dir_binary` image, I can get a much cleaner representation of the lane lines.

Finally, I mask the binary image to remove all pixels outside the relevant area of the image that contains the road (defined in lines 240-244).

Here's an example of my output for this step.

![alt text][image3]

3\. **Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.**

The code for my perspective transform includes a function called `perspective_transform`, which appears in lines 1 through 8.  The `perspective_transform` function takes as inputs an image (`img`), as well as the transform matrix required to perform the perspective transform (`M`). Since this matrix remains the same throughout all the computations, it is created as a global variable `transform_M` on lines 548-564 (along with the inverse matrix `inverse_M`) using pre-defined source (`undist_src`) and destination (`undist_dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
top_left = [632, 425]
top_right = [648, 425]
bottom_right = [1080, 710]
bottom_left = [210, 710]
undist_src = np.float32([top_left,top_right,bottom_right,bottom_left])

w = img_shape[0] # 1280
h = img_shape[1] # 720
xp = 300 # x padding offset
topp = -3000 # top padding offset
botp = 0 # bottom padding offset
undist_dst = np.float32([[xp,topp],[w-xp,topp],[w-xp,h-botp],[xp,h-botp]])

```
This resulted in the following source and destination points:

| Corner          | Source        | Destination   | 
|:---------------:|:--------------:|:---------------:| 
| top left          | 632, 425      | 300, -3000    | 
| top right        | 648, 425      | 980, -3000    |
| bottom right  | 1080, 710    | 980, 720       |
| bottom left    | 210, 710      | 300, 720       |

**Note**: *The -3000 y-value for the top destination points may seem strange, but this is due to the fact that the top source points are so far up and close together. This large negative y-value stretches these points far upward so that the lanes become a rectangle in the transformed image.*

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

4\. **Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

In the image processing pipeline, I keep track of each lane line as an object in python using the Line class defined on lines 18-77 in my code. In each call to the `pipeline` function, after undistorting, binarizing and warping the image frame, I decide how to proceed with detecting the lane lines. If it's the first frame being processed, or if the lines were not successfully detected in the previous frame (due to either insufficient gradient/color data, or failing the sanity check), then I perform a sliding window search across the entire image by calling the `find_lanes_sliding_window` function on line 597. Otherwise, if the lines were successfully detected in the previous frame and they pass the sanity check, I perform a limited search near where the lines were last detected by calling the `find_lanes_limited_search` function on line 600, in order to save on performance time.

The sanity check is defined at lines 452-467, and it ensures a couple things: that the width of each lane is reasonably close to the expected lane width, and that the curve radius is similar for both lines. As long as the sanity check succeeds, we continue using the limited search function. If the sanity check fails at any point, then we go back to using the more expensive sliding window search.

The `find_lanes_sliding_window` function is defined at lines 264-364. This function assumes that we have no idea where in the image the lines will appear. So first, a histogram is created to measure the number of pixels in the bottom half of the binary image (lines 266-274). The two x-positions with the most pixels (the peaks of the histogram) are used as the starting point for the left and right lines. Then, a sliding window is used to follow the lines up to the top of the image (lines 276-317). Finally, I fit the lane lines with a 2nd order polynomial.

The `find_lanes_limited_search` function is defined at lines 366-450. Instead of starting with no information, this function uses the location of the lane lines in the previous frame to create a limited search area for the lines in the current frame. Specifically, it searches within a margin of 100 pixels to the left and right of the previous lines (code lines 368-378). Finally, I fit the lane lines with a 2nd order polynomial.

Using either function, the resulting polynomials look like this:

![alt text][image5]

5\. **Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.**

In both of the lane-finding search functions, I call the `meter_curve_radius` function defined at lines 477-488 for calculating the radius of curvature in meters. This function uses several pre-defined scalers to convert the polynomials for each line into world-space dimensions. These scalers are hard-coded as follows in lines 15 and 16:

```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

```

Once the polynomials are in world-space, the radius is computed at a specific y-position of each polynomial. I used the bottom of the image as the y-position.

The position of the vehicle is calculated at lines 350-355 (and again at lines 432-437), by subtracting the center of the image from the "true" vehicle center, and multiplying the result by the conversion scaler `xm_per_pix`. The true vehicle center is computed by halving the distance between the two lane lines at the bottom of the image.

6\. **Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.**

I implemented this step in lines 490-532 in my code in the `draw_lane` function. The image is unwarped using the inverse transformation matrix mentioned in step 3, and the lane is drawn onto the image. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

1\. **Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).**

Here's a [link to my video result](project_solution.mp4).

---

### Discussion

1\. **Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The approach I took with this project was to focus on creating a great binary image output, composed of the best combinations of gradients and color information, so that it would be less prone to making mistakes, and I would have to do less work making corrections to the line detections later on. Overall, this approach seemed to work well. The two big game-changing revelations I made in this approach were: (1) deciding to use erosion and dilation of pixels in the directional binary output, and (2) using a positional mask to get rid of all the irrelevant pixel data in the binary output. Both of these changes made my algorithm much more robust and predictable.

One issue I faced was handling cases when the sanity check failed. Initially, whenever the sanity check failed after a limited search, it would immediate conduct a full-image window search for the lines on the same frame and save these resulting lines into their respective Line objects, regardless of whether this second search for the lane lines passed the sanity check or not. The problem was the lines produced by the second search frequently failed the sanity check, so my Line objects were constantly being "corrupted" with false-positives, resulting in lots of eratic lane lines in the video. To remedy this, I decided that if the lines still did not pass the sanity check on the second full-image search, then the algorithm would simply throw out these lines altogether, and instead draw the lines produced from the average of the past 10 successful frames. This worked much better, and solved most of the issues with eratic lines.

The algorithm could fail if the lane appears in a different part of the image entirely for some reason, since I am masking and warping the image based on an assumption about where the lanes will appear. It could also produce incorrect results if the lane lines on the road disappeared entirely (perhaps due to snow or worn-out paint on the road). It would still attempt to draw the lanes produced by the 10 previous successful frames, but if the road changed at all after the lanes became invisible, this algorithm would have no way of knowing.

If I were to pursue this project further, I would attempt to remedy this problem caused by the edge case of invisible lane lines, by perhaps trying to detect signs on the sides of the road, or other cars in other lanes, in order to estimate where the lanes might be.
