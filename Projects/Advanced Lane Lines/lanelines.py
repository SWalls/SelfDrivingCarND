import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Number of past iterations to store in a Line.
n = 10
# Image shape of all lane lines images.
img_shape = (1280,720)
# Whether to save images from cv pipeline
TEST_PIPELINE = True
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n fits of the line
        self.recent_fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    # polynomial coefficients for the most recent fit
    def current_fit(self):
        return self.recent_fits[0]

    def add_fit_x_vals(self, fitx):
        global n
        if len(self.recent_xfitted) < n:
            self.recent_xfitted.append(fitx)
        history_lim = min(n, len(self.recent_xfitted))
        for i in range(history_lim-1,0,-1):
            self.recent_xfitted[i] = self.recent_xfitted[i-1]
        self.recent_xfitted[0] = fitx
        self.bestx = np.mean(self.recent_xfitted, axis=0)

    def add_fit(self, fit):
        global n
        if len(self.recent_fits) < n:
            self.recent_fits.append(fit)
        history_lim = min(n, len(self.recent_fits))
        for i in range(history_lim-1,0,-1):
            self.recent_fits[i] = self.recent_fits[i-1]
        self.recent_fits[0] = fit
        self.best_fit = np.mean(self.recent_fits, axis=0)
        if len(self.recent_fits) > 1:
            self.diffs = self.recent_fits[0] - self.recent_fits[1]

    def update(self, fit, fitx, curverad, vcenter, x, y):
        global img_shape, xm_per_pix
        h = img_shape[1]
        self.detected = True
        self.add_fit_x_vals(fitx)
        self.add_fit(fit)
        self.radius_of_curvature = curverad
        self.line_base_pos = (vcenter - fitx[h-1])*xm_per_pix
        self.allx = x
        self.ally = y

def plot_two(img1, img2, title1, title2, fontsize=50):
    plt.clf()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    ax1.imshow(img1_rgb)
    ax1.set_title(title1, fontsize=fontsize)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    ax2.imshow(img2_rgb)
    ax2.set_title(title2, fontsize=fontsize)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Calibrate camera using same image shape for all images.
def calibrate_camera(img_shape):
    # Chessboard dimensions
    nx = 9
    ny = 6

    # Read in and make a list of calibration images
    images = glob.glob("camera_cal/calibration*.jpg")

    # Arrays to store object points and image points from all the calibration images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Prepare object points
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates

    for i in range(len(images)):
        # Read in each image
        fname = images[i]
        img = mpimg.imread(fname)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If corners are found, add object points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display corners
            # img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # cv2.imwrite("output_images/chessboard-corners-%d.jpg" % i, img)

    # Calibrate camera
    cam_ret, cam_mtx, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera( \
        objpoints, imgpoints, img_shape, None, None)
    
    return cam_mtx, cam_dist

# Undistort image.
def undistort(img, cam_mtx, cam_dist):
    # Undistort using camera matrix and distortion coefficients.
    undist = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
    return undist

# Perspective transform (warp) image.
def perspective_transform(img, M):
    # Warp the image to a top-down view
    shape = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, \
        shape, flags=cv2.INTER_LINEAR)
    return warped

# Applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, \
                1 if orient == 'x' else 0, \
                1 if orient == 'y' else 0, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

# Applies Sobel x and y, then computes the magnitude of the gradient
# and applies a threshold.
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    mag_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobelxy/np.max(mag_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary output image
    return sbinary

# Applies Sobel x and y, then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(abs_sobely)
    sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

# Computes a channel of the HLS color space and applies a threshold.
def hls_thresh(img, channel='s', thresh=(170, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    c = 0 if channel == 'h' else (1 if channel == 'l' else 2)
    hls_channel = hls[:,:,c]
    # Threshold color channel
    sbinary = np.zeros_like(hls_channel)
    sbinary[(hls_channel >= thresh[0]) & (hls_channel <= thresh[1])] = 1
    return sbinary

def binarize(img_filename, undist, cam_mtx, cam_dist):
    global transform_M, TEST_PIPELINE

    # Sobel kernel size
    ksize = 5

    # Apply each of the thresholding functions
    gradx_binary = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # cv2.imwrite("output_images/%s_sobelx.jpg" % img_filename, gradx_binary*255)
    grady_binary = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    # cv2.imwrite("output_images/%s_sobely.jpg" % img_filename, grady_binary*255)
    mag_binary = mag_thresh(undist, sobel_kernel=ksize, thresh=(30, 100))
    # cv2.imwrite("output_images/%s_mag.jpg" % img_filename, mag_binary*255)
    dir_binary = dir_thresh(undist, sobel_kernel=ksize, thresh=(0.8, 1.2))
    # Erode the direction results so that only big globs of nearby white pixels survive.
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dir_binary = cv2.erode(dir_binary, erode_kernel, iterations=1)
    dir_binary = cv2.dilate(dir_binary, erode_kernel, iterations=2)
    # cv2.imwrite("output_images/%s_dir.jpg" % img_filename, dir_binary*255)
    hls_binary = hls_thresh(undist, channel='s', thresh=(170, 255))
    # cv2.imwrite("output_images/%s_hls.jpg" % img_filename, hls_binary*255)

    # Combine thresholds
    combined = np.zeros_like(mag_binary)
    combined[((gradx_binary == 1) & (grady_binary == 1)) | \
            ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    # cv2.imwrite("output_images/%s_thresh.jpg" % img_filename, combined*255)

    # Mask: exclude pixels outside lanes window.
    mask_top_left = [590, 425]
    mask_top_right = [690, 425]
    mask_bottom_right = [1200, 710]
    mask_bottom_left = [210, 710]
    mask = np.float32([mask_top_left,mask_top_right,mask_bottom_right,mask_bottom_left])
    points = []
    for i in range(len(mask)):
        points.append([mask[i]])
        points.append([mask[(i+1)%len(mask)]])
    poly = np.array(points, dtype=np.int32)
    matrix = np.zeros(combined.shape, dtype=np.int32)
    cv2.drawContours(matrix, [poly] , -1, (1), thickness=-1)
    masked = np.zeros_like(combined)
    masked[(combined == 1) & (matrix == 1)] = 1
    if TEST_PIPELINE:
        cv2.imwrite("output_images/%s_combo_masked.jpg" % img_filename, masked*255)

    # Do perspective transform of masked thresholded image.
    binary_warped = perspective_transform(masked, transform_M)
    # cv2.imwrite("output_images/%s_persp_thresh.jpg" % img_filename, binary_warped*255)

    return binary_warped

def find_lanes_sliding_window(img_filename, binary_warped, left_line, right_line):
    global img_shape, xm_per_pix
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualize: Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.clf()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if TEST_PIPELINE:
        plt.savefig("output_images/%s_polylanes_vis.jpg" % img_filename)

    # Calculate the vehicle center.
    y_eval = np.max(ploty)
    vehicle_center = (left_fitx[y_eval] + right_fitx[y_eval]) / 2
    img_center = (img_shape[0]/2.)
    # print ("Img center: %.2f, Vehicle center: %.2f" % (img_center, vehicle_center))
    vehicle_center_offset_m = (vehicle_center - img_center)*xm_per_pix

    # Get radius of curvature in meters.
    left_curverad, right_curverad = meter_curve_radius(left_fit, right_fit, ploty, left_fitx, right_fitx)

    # Update lines definitions.
    left_line.update(left_fit, left_fitx, left_curverad, vehicle_center, leftx, lefty)
    right_line.update(right_fit, right_fitx, right_curverad, vehicle_center, rightx, righty)

    return ploty, vehicle_center_offset_m

def find_lanes_limited_search(img_filename, binary_warped, left_line, right_line):
    global img_shape, xm_per_pix
    # We now have a new warped binary image 
    # from the next frame of video.
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Generate y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # Return if there has been an error.
    if lefty.size == 0 or leftx.size == 0 or righty.size == 0 or rightx.size == 0:
        print ("Error: Did not detect a lane in this frame.")
        return ploty, 0

    # Fit a second order polynomial to each line.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Visualize: Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.clf()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if TEST_PIPELINE:
        plt.savefig("output_images/%s_polylanes_vis_lim.jpg" % img_filename)

    # Calculate the vehicle center.
    y_eval = np.max(ploty)
    vehicle_center = (left_fitx[y_eval] + right_fitx[y_eval]) / 2
    img_center = (img_shape[0]/2.)
    # print ("Img center: %.2f, Vehicle center: %.2f" % (img_center, vehicle_center))
    vehicle_center_offset_m = (vehicle_center - img_center)*xm_per_pix

    # Get radius of curvature in meters.
    left_curverad, right_curverad = meter_curve_radius(left_fit, right_fit, ploty, left_fitx, right_fitx)

    if sanity_check(left_fitx, right_fitx, left_curverad, right_curverad):
        # Update lines definitions.
        left_line.update(left_fit, left_fitx, left_curverad, vehicle_center, leftx, lefty)
        right_line.update(right_fit, right_fitx, right_curverad, vehicle_center, rightx, righty)
    else:
        left_line.detected = False
        right_line.detected = False

    return ploty, vehicle_center_offset_m

def sanity_check(left_fitx, right_fitx, left_curverad, right_curverad):
    global img_shape, xm_per_pix, xp
    y_eval = img_shape[1] - 1
    # Make sure lane width is reasonably close to expected lane width.
    lane_width = (right_fitx[y_eval] - left_fitx[y_eval])
    expected_width = img_shape[0]-(xp*2)
    width_diff = abs(1. - (lane_width / expected_width))
    # print ("Width difference: %.2f" % width_diff)
    if width_diff > 0.1:
        return False
    # Make sure curve radius is similar for both lines.
    curverad_diff = abs(1. - (left_curverad / right_curverad))
    # print ("Curve radius difference: %.2f" % curverad_diff)
    if curverad_diff > 1:
        return False
    return True

def pixel_curve_radius(left_fit, right_fit, ploty):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return left_curverad, right_curverad

def meter_curve_radius(left_fit, right_fit, ploty, leftx, rightx):
    global ym_per_pix, xm_per_pix
    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty) # bottom of image
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad

def draw_lane(img_filename, undist, binary_warped, left_line, right_line, vcenteroffset, ploty):
    global inverse_M

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_lane_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_lines_warp = color_lane_warp.copy()

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_lane_warp, np.int_([pts]), (0, 255, 0))
    
    # Color in left and right line pixels
    color_lines_warp[left_line.ally, left_line.allx] = [255, 0, 0]
    color_lines_warp[right_line.ally, right_line.allx] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix
    lane_unwarp = perspective_transform(color_lane_warp, inverse_M)
    lines_unwarp = perspective_transform(color_lines_warp, inverse_M)
    binary_lines = lines_unwarp.copy()
    nonzero = np.nonzero(binary_lines.any(axis=-1))
    binary_lines[nonzero] = [255, 255, 255]
    # Combine the result with the original image
    lane = cv2.addWeighted(undist, 1, lane_unwarp, 0.3, 0)
    dampened = cv2.addWeighted(lane, 1, binary_lines, -0.9, 0)
    result = cv2.addWeighted(dampened, 1, lines_unwarp, 0.5, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    avg_curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature)/2.
    rad_text = "Radius of curvature: %.2f m" % avg_curvature
    cv2.putText(result, rad_text, (10,40), font, 1, (255,255,255), 2, cv2.LINE_AA)
    offset_text = "Vehicle is %.2f m %s of center" % \
        (abs(vcenteroffset), "right" if vcenteroffset >= 0 else "left")
    cv2.putText(result, offset_text, (10,80), font, 1, (255,255,255), 2, cv2.LINE_AA)

    if TEST_PIPELINE:
        cv2.imwrite("output_images/%s_output.jpg" % img_filename, result)

    return result

if TEST_PIPELINE:
    img_filename = "test1"
else:
    img_filename = ""

# Get camera matrix and distortion coefficients.
cam_mtx, cam_dist = calibrate_camera(img_shape)

if TEST_PIPELINE:
    image = cv2.imread("camera_cal/calibration1.jpg")
    undist = undistort(image, cam_mtx, cam_dist)
    plot_two(image, undist, "Original image", "Undistorted image")
    plt.savefig("output_images/undistort_chessboard.jpg")

# Predefine src and dst points (after undistortion) for 
# perspective transform of each lane lines image.
top_left = [632, 425]
top_right = [648, 425]
bottom_right = [1080, 710]
bottom_left = [210, 710]
undist_src = np.float32([top_left,top_right,bottom_right,bottom_left])
w = img_shape[0]
h = img_shape[1]
xp = 300 # x padding offset
topp = -3000 # top padding offset
botp = 0 # bottom padding offset
undist_dst = np.float32([[xp,topp],[w-xp,topp],[w-xp,h-botp],[xp,h-botp]])

# Get the transform matrix, using src and dst points.
transform_M = cv2.getPerspectiveTransform(undist_src, undist_dst)
inverse_M = cv2.getPerspectiveTransform(undist_dst, undist_src)

# Create lines.
left_line = Line()
right_line = Line()

def pipeline(image):
    global cam_mtx, cam_dist, left_line, right_line, transform_M, img_filename
    global undist_src, undist_dst

    if not TEST_PIPELINE:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert to BGR

    undist = undistort(image, cam_mtx, cam_dist)

    if TEST_PIPELINE:
        undist_copy = undist.copy()
        src_pts = undist_src.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(undist_copy, [src_pts], True, (255,0,0), 5)
        warped = perspective_transform(undist, transform_M)
        dst_pts = undist_dst.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(warped, [dst_pts], True, (255,0,0), 5)
        plot_two(undist_copy, warped, \
            "Undistorted image with src points drawn", \
            "Warped result with dest. points drawn", 30)
        plt.savefig("output_images/%s_warped.jpg" % img_filename)
        cv2.imwrite("output_images/%s_undist.jpg" % img_filename, undist)

    # Get binary image of undistorted and unwarped lanes after thresholding.
    binary_warped = binarize(img_filename, undist, cam_mtx, cam_dist)

    # Find the lane lines and populate Line member vars.
    if not left_line.detected:
        ploty, vcenteroffset = find_lanes_sliding_window(img_filename, \
            binary_warped, left_line, right_line)
    else:
        ploty, vcenteroffset = find_lanes_limited_search(img_filename, \
            binary_warped, left_line, right_line)
        # Re-search if it failed to detect lines.
        if not left_line.detected:
            print ("Sanity check failed!")
            # ploty, vcenteroffset = find_lanes_sliding_window(img_filename, \
            #    binary_warped, left_line, right_line)

    # Draw the lane back onto the original image.
    result = draw_lane(img_filename, undist, binary_warped, left_line, right_line, vcenteroffset, ploty)

    if not TEST_PIPELINE:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) # convert back to RGB

    return result

# Read image from file, and undistort.
if TEST_PIPELINE:
    image = cv2.imread("test_images/%s.jpg" % img_filename)
    pipeline(image)
else:
    soln_output = 'project_solution.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    soln_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    soln_clip.write_videofile(soln_output, audio=False)
