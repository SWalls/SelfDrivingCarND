import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os.path
from random import randint
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

def convert_color(image, color_space):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(image)
    return feature_image

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = convert_color(image, color_space)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def train_model(svc_model_fname, color_space, spatial_size, hist_bins, orient, 
                pix_per_cell, cell_per_block, hog_channel):
    print("Loading training data.")

    # Divide up into cars and notcars
    cars = glob.glob('training_data/vehicles/KITTI_extracted/*.png')
    cars.extend(glob.glob('training_data/vehicles/GTI_Far/*.png'))
    cars.extend(glob.glob('training_data/vehicles/GTI_Left/*.png'))
    cars.extend(glob.glob('training_data/vehicles/GTI_MiddleClose/*.png'))
    cars.extend(glob.glob('training_data/vehicles/GTI_Right/*.png'))
    notcars = glob.glob('training_data/non-vehicles/Extras/*.png')
    notcars.extend(glob.glob('training_data/non-vehicles/GTI/*.png'))

    print("Extracting HOG features.")

    t=time.time()
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Save the SVC model.
    dist_pickle = dict()
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    joblib.dump(dist_pickle, svc_model_fname)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return dist_pickle

# Either load the model from the disk or train it.
def get_classifier():
    svc_model_fname = 'svc_pickel.p'
    if os.path.isfile(svc_model_fname):
        # load the model
        dict = joblib.load(svc_model_fname)
    else:
        # train the model
        dict = train_model(svc_model_fname, color_space, spatial_size, hist_bins, 
            orient, pix_per_cell, cell_per_block, hog_channel)
    svc = dict["svc"]
    X_scaler = dict["scaler"]
    return svc, X_scaler

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, color_space, orient, 
        pix_per_cell, cell_per_block, spatial_size, hist_bins):
    global decision_threshold

    img_norm = img.astype(np.float32)/255
    
    img_tosearch = img_norm[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Sliding window search!
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            # test_prediction = svc.predict(test_features)
            test_decision = svc.decision_function(test_features)[0]
            
            if test_decision > decision_threshold:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox = ((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart))
                bboxes.append(bbox)

    return bboxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def flatten(heatmap):
    # Make all heats = 1
    heatmap[heatmap > 0] = 1
    # Return flattened map
    return heatmap

# Define a function to draw bounding boxes
def draw_bboxes(img, bboxes, color=(0, 255, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Draw bounding boxes produced from labels
def draw_labeled_bboxes(img, labels, color=(0, 255, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through all bounding boxes
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Do sanity check on bounding box.
        # Make sure aspect ratio makes sense.
        bb_width = bbox[1][0]-bbox[0][0]
        bb_height = bbox[1][1]-bbox[0][1]
        bb_aspect_ratio = bb_width/bb_height
        if bb_aspect_ratio > 2.3 or bb_aspect_ratio < 0.8:
            continue
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image
    return imcopy

# Draw two images side-by-side with pyplot
def plot_two(img1, img2, title1, title2, fontsize=50):
    plt.clf()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=fontsize)
    ax2.imshow(img2, cmap='hot')
    ax2.set_title(title2, fontsize=fontsize)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

class Frame: # Contains information for one frame of the video.
    def __init__(self, bbox_list):
        global img_height, img_width
        global heat_threshold
        # Init heatmap.
        self.heatmap = np.zeros((img_height, img_width))
        # Add heat to heatmap.
        self.heatmap = add_heat(self.heatmap, bbox_list)
        # Apply threshold to remove false positives.
        self.heatmap = apply_threshold(self.heatmap, heat_threshold)
        # Create flattened heatmap.
        self.flatmap = flatten(self.heatmap)
        '''
        # Create labels from heatmap.
        self.labels = label(self.heatmap)
        # Find the peaks of the heatmap; one for each label.
        # The peaks are used to tell when a car has been detected 
        # for multiple frames in the past.
        self.peak_coords = peak_local_max(self.heatmap, min_distance=20, 
                labels=self.labels[0], num_peaks_per_label=1)
        '''

num_frames = 20 # Duration (in frames) of heatmaps to keep
frames = deque(maxlen=num_frames) # Queue of previous frames
current_frame = 0 # Current frame in video

# The main pipeline
def pipeline(img):
    global frames, frame_threshold, num_frames, current_frame
    global svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins
    # Find cars in image.
    bbox_list = []
    # Gather bounding boxes at different scales, using different search boundaries for each scale.
    for scale, ystart, ystop, xstart, xstop in [[1.5, 400, 656, 0, img_width], [2.5, 380, 656, img_width-300, img_width]]:
        bbox_list.extend(find_cars(img, ystart, ystop, xstart, xstop, scale, svc, 
            X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    # Use bounding boxes to create new frame with heatmap.
    frame = Frame(bbox_list)
    # Combine the flattened heatmaps from all previous frames to create a time-based heatmap.
    timemap = np.sum(np.array([f.flatmap for f in frames]), axis=0)
    # Weight the heatmap from the current frame more than the previous frames.
    timemap = frame.flatmap*current_frame_weight*num_frames + timemap*(1-current_frame_weight)*num_frames
    timemap = timemap/num_frames
    # Add the current frame to the list of previous frames.
    frames.append(frame)
    # Apply threshold to timemap to help remove false positives
    timemap = apply_threshold(timemap, frame_threshold)
    # Visualize the heatmap when displaying
    timemap = np.clip(timemap, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(timemap)
    # Draw boxes onto the image.
    draw_label_img = draw_labeled_bboxes(img, labels)
    # Optionally save the video frame and heatmap.
    if SAVE_VIDEO_FRAMES:
        plot_two(img, timemap, ("Video Frame %d" % current_frame), "Heatmap")
        plt.savefig('output_images/video_frames/frame-%d.png' % current_frame)
        # Optionally save images of a frame's labels and bounding boxes
        if current_frame == SPECIAL_VIDEO_FRAME:
            plt.clf()
            plt.imshow(labels[0], cmap='gray')
            plt.savefig('output_images/video_frames/labels-%d.png' % current_frame)
            plt.clf()
            plt.imshow(draw_label_img)
            plt.savefig('output_images/video_frames/bboxes-%d.png' % current_frame)
        current_frame += 1
    if TEST_PIPELINE:
        draw_boxes_img = draw_bboxes(img, bbox_list)
        return draw_label_img, draw_boxes_img, timemap
    else:
        return draw_label_img

# Set up the parameters for HOG feature extraction, spatial binning, and color histograms.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 16
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

# Define image dimensions
img_width = 1280
img_height = 720

TEST_HOG = False # If True, will test HOG output on test images
TEST_PIPELINE = False # If True, will test pipeline on test images
SAVE_VIDEO_FRAMES = True # If True, will save video frames with heatmaps
SPECIAL_VIDEO_FRAME = 29 # If SAVE_VIDEO_FRAMES is True, will save this frame's bbox and labels

# Set up thresholds.
heat_threshold = 0 # Minimum heatmap number required to justify a detection in one frame
decision_threshold = 0 # Minimum value for classifier decisions
current_frame_weight = 0.3 # How much to weight the current frame over the previous frames
frame_threshold = 8 # Minimum heatmap frame detections required for true positive detection
if TEST_PIPELINE:
    frame_threshold = 0

# Get the classifier.
svc, X_scaler = get_classifier()

if TEST_HOG:
    # Save hog output on random image from each class (car, non-car).
    i = 0
    for fname in ["training_data/vehicles/KITTI_extracted/%d.png", 
                    "training_data/non-vehicles/GTI/image%d.png"]:
        fnum = randint(1,1000)
        img = mpimg.imread(fname % fnum)
        img = img.astype(np.float32)/255
        f, hog_img = get_hog_features(img[:,:,0], orient, pix_per_cell, 
                                    cell_per_block, vis=True, feature_vec=True)
        plt.clf()
        plt.imshow(hog_img)
        plt.savefig('output_images/hog-class%d-img%d-%s-orient%d-ppc%d-cpb%d.png' % 
            (i, fnum, color_space, orient, pix_per_cell, cell_per_block))
        i += 1
elif TEST_PIPELINE:
    # Save images produced by pipeline using test images as input.
    for i in range(1, 7):
        frames.clear()
        img = mpimg.imread('test_images/test%d.jpg' % i)
        draw_label_img, draw_boxes_img, heatmap = pipeline(img)
        plt.clf()
        plt.imshow(heatmap, cmap='hot')
        # plt.plot(peak_coords[:,1], peak_coords[:,0], 'r.')
        plt.savefig('output_images/test%d-heatmap.png' % i)
        plt.clf()
        plt.imshow(draw_boxes_img)
        plt.savefig('output_images/test%d-bboxes.png' % i)
        plt.clf()
        plt.imshow(draw_label_img)
        plt.savefig('output_images/test%d-out.png' % i)
else:
    # Run pipeline over video.
    soln_output = 'project_solution.mp4'
    clip1 = VideoFileClip("project_video.mp4").subclip(29,31) #.subclip(4,14) #.subclip(36,43)
    soln_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    soln_clip.write_videofile(soln_output, audio=False)