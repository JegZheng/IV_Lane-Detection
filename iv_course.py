import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy.misc import imresize

def process_image(ori,image,img_N):
	# Read in and grayscale the image
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Define our parameters for Canny and apply
	low_threshold = 10
	high_threshold = 60
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)
	ignore_mask_color = 255

	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]),(400, 250), (800, 250), (imshape[1],imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)
	#plt.imshow(mask)
	#plt.show()


	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 2    # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 10 #minimum number of pixels making up a line
	max_line_gap = 2    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
		                min_line_length, max_line_gap)

	#cluster all the lines to two lines


	lines_left = []
	lines_right = []
	for line in lines:
		for x1, y1, x2, y2 in line:
			if x2 - x1 == 0: continue;  # Infinite slope
			slope = float(y2-y1)/float(x2-x1)
			if abs(slope) < .5 or abs(slope) > .9: continue  # Discard unlikely slopes
			if slope > 0: 
				lines_left += [(x1,y1),(x2,y2)]
			else: 
				lines_right += [(x1,y1),(x2,y2)]

	left_xs = list(map(lambda x: x[0], lines_left))
	left_ys = list(map(lambda x: x[1], lines_left))
	right_xs = list(map(lambda x: x[0], lines_right))
	right_ys = list(map(lambda x: x[1], lines_right))

	#left_fit = np.poly1d(np.polyfit(left_xs, left_ys, 2))
	#right_fit = np.poly1d(np.polyfit(right_xs, right_ys, 2))

	left_fit = np.polyfit(left_xs, left_ys, 1)
	right_fit = np.polyfit(right_xs, right_ys, 1)

	y1 = image.shape[0] # Bottom of image
	y2 = image.shape[0] / 2 + 50 # Middle of view
	x1_left = (y1 - left_fit[1]) / left_fit[0]
	x2_left = (y2 - left_fit[1]) / left_fit[0]
	x1_right = (y1 - right_fit[1]) / right_fit[0]
	x2_right = (y2 - right_fit[1]) / right_fit[0]    
	y1 = int(y1); y2 = int(y2);
	x1_left = int(x1_left); x2_left = int(x2_left);
	x1_right = int(x1_right); x2_right = int(x2_right);
	cv2.line(line_image, (x1_left, y1), (x2_left, y2), (255, 0, 0), 5)
	cv2.line(line_image, (x1_right, y1), (x2_right, y2), (255, 0, 0), 5)



	line = np.array([[x1_right, y1], [x2_right, y2],[x1_left, y1],[x2_left, y2]])

	# Draw the lines on the edge image

	#ori = imresize(ori, (720, 1280, 3))
	if line_image.shape != ori.shape:
		line_image = imresize(line_image,ori.shape)
	image_result = cv2.addWeighted(ori,0.8,line_image,1,0)
	print line
	plt.imshow(image_result)
	plt.savefig('lane_'+img_N)

	return line

if __name__ == '__main__':
	for i in range(6):
		image_name="test" + str(i+1)+ ".jpg"
		img_pre = "predic_" + str(i+1)+ ".jpg"
		ori = mpimg.imread(image_name)
		image = mpimg.imread(img_pre)
		line = process_image(ori,image,image_name)
		with open('vertice.txt','a') as o:
			for i in line:
				for j in i:
					o.write(str(j)+' ')
			o.write('\n')

