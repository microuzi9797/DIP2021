import cv2
import numpy
import math
from scipy import signal

# Problem 1: EDGE DETECTION
img1 = cv2.imread("sample1.jpg", cv2.IMREAD_GRAYSCALE)
img1_height = img1.shape[0]
img1_width = img1.shape[1]
img2 = cv2.imread("sample2.jpg", cv2.IMREAD_GRAYSCALE)
img2_height = img2.shape[0]
img2_width = img2.shape[1]
# 1st order edge detection
img1_first_order = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
b1 = 1
threshold = 20
row_filter = numpy.array([[-1, 0, 1], [-b1, 0, b1], [-1, 0, 1]])
col_filter = numpy.array([[-1, -b1, -1], [0, 0, 0], [1, b1, 1]])
for i in range(img1_height):
	for j in range(img1_width):
		row_magnitude = 0
		col_magnitude = 0
		for k in range(3):
			for l in range(3):
				if(i + k - 1 >= 0 and i + k - 1 < img1_height and j + l - 1 >= 0 and j + l - 1 < img1_width):
					row_magnitude += int(img1[i + k - 1][j + l - 1]) * row_filter[k][l]
					col_magnitude += int(img1[i + k - 1][j + l - 1]) * col_filter[k][l]
				else:
					row_magnitude += int(img1[i][j]) * row_filter[k][l]
					col_magnitude += int(img1[i][j]) * col_filter[k][l]
		magnitude = math.sqrt(math.pow(row_magnitude, 2) + math.pow(col_magnitude, 2))
		if(magnitude > threshold * (b1 + 2)):
			img1_first_order[i][j] = 255
		else:
			img1_first_order[i][j] = 0
cv2.imwrite("result1.jpg", img1_first_order)
print("Problem 1 (a) (1) done.")
# 2nd order edge detection
img1_second_order_first = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
img1_second_order = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
high_threshold = 100
low_threshold = 20
kernel = numpy.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]], dtype = 'uint8') / 273
img1_second_filter = signal.convolve2d(img1, kernel, mode = "same")
for i in range(img1_height):
	for j in range(img1_width):
		if(img1_second_filter[i][j] >= high_threshold):
			img1_second_order_first[i][j] = 0
		elif(img1_second_filter[i][j] >= low_threshold):
			img1_second_order_first[i][j] = -1
		else:
			img1_second_order_first[i][j] = 1
for i in range(2,img1_height - 2):
	for j in range(2,img1_width - 2):
		if(img1_second_order_first[i][j] == 0):
			if(numpy.abs(img1_second_order_first[i - 1][j - 1]) + numpy.abs(img1_second_order_first[i + 1][j + 1]) > 0 and numpy.abs(img1_second_order_first[i - 1][j]) + numpy.abs(img1_second_order_first[i + 1][j]) > 0  and numpy.abs(img1_second_order_first[i - 1][j + 1]) + numpy.abs(img1_second_order_first[i + 1][j - 1]) > 0 and numpy.abs(img1_second_order_first[i][j - 1]) + numpy.abs(img1_second_order_first[i][j + 1]) > 0):
				img1_second_order[i][j] = 255
			else:
				img1_second_order[i][j] = 0
		else:
			img1_second_order[i][j] = 0
cv2.imwrite("result2.jpg", img1_second_order)
print("Problem 1 (a) (2) done.")
# Canny edge detection
# noise reduction: σ = 1.4
def noise_reduction(img):
	Gaussian_filter = numpy.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159
	img_noise_filter = signal.convolve2d(img, Gaussian_filter, mode = "same")
	return img_noise_filter
# compute gradient magnitude and orientation
def compute_gradient(img, b):
	img_height = img.shape[0]
	img_width = img.shape[1]
	row_filter = numpy.array([[-1, 0, 1], [-b, 0, b], [-1, 0, 1]])
	col_filter = numpy.array([[-1, -b, -1], [0, 0, 0], [1, b, 1]])
	row_magnitude = numpy.zeros((img_height, img_width))
	col_magnitude = numpy.zeros((img_height, img_width))
	for i in range(img_height):
		for j in range(img_width):
			row_sum = 0
			col_sum = 0
			for k in range(3):
				for l in range(3):
					if(i + k - 1 >= 0 and i + k - 1 < img_height and j + l - 1 >= 0 and j + l - 1 < img_width):
						row_sum += int(img[i + k - 1][j + l - 1]) * row_filter[k][l]
						col_sum += int(img[i + k - 1][j + l - 1]) * col_filter[k][l]
					else:
						row_sum += int(img[i][j]) * row_filter[k][l]
						col_sum += int(img[i][j]) * col_filter[k][l]
			row_magnitude[i][j] = row_sum
			col_magnitude[i][j] = col_sum
	img_gradient = numpy.hypot(row_magnitude, col_magnitude)
	img_gradient = (img_gradient / img_gradient.max()) * 255
	theta = numpy.arctan2(col_magnitude, row_magnitude)
	return img_gradient, theta
# non-maximal suppression
def non_maximal_suppression(img, theta):
	img_height = img.shape[0]
	img_width = img.shape[1]
	img_non_maximal = numpy.zeros((img_height, img_width), dtype = 'uint8')
	angle = theta * 180.0 / math.pi
	for i in range(1, img_height - 1):
		for j in range(1, img_width - 1):
			G1 = 0
			G2 = 0
			# 0 <= angle <= 180
			if(angle[i][j] < 0):
				angle[i][j] += 180.0
			# angle == 0
			if((0 <= angle[i][j] < 22.5) or (157.5 < angle[i][j] <= 180)):
				G1 = img[i][j + 1]
				G2 = img[i][j - 1]
			# angle == 45
			elif(22.5 <= angle[i][j] <= 67.5):
				G1 = img[i + 1][j - 1]
				G2 = img[i - 1][j + 1]
			# angle == 90
			elif(67.5 < angle[i][j] < 112.5):
				G1 = img[i + 1][j]
				G2 = img[i - 1][j]
			# angle == 135
			elif(112.5 <= angle[i][j] <= 157.5):
				G1 = img[i - 1][j - 1]
				G2 = img[i + 1][j + 1]
			# decision
			if(img[i][j] >= G1) and (img[i][j] >= G2):
				img_non_maximal[i][j] = img[i][j]
			else:
				img_non_maximal[i][j] = 0
	return img_non_maximal
# hysteretic thresholding
def hysteretic_thresholding(img, threshold_H, threshold_L):
	img_height = img.shape[0]
	img_width = img.shape[1]
	img_thresholding = numpy.zeros((img_height, img_width), dtype = 'uint8')
	for i in range(img_height):
		for j in range(img_width):
			if(img[i][j] >= threshold_H):
				img_thresholding[i][j] = 1
			elif(img[i][j] >= threshold_L):
				img_thresholding[i][j] = 0
			else:
				img_thresholding[i][j] = -1
	return img_thresholding
# connected component labeling method
def connected_component_labeling(img):
	img_height = img.shape[0]
	img_width = img.shape[1]
	img_label = numpy.zeros((img_height, img_width), dtype = 'uint8')
	for i in range(img_height):
		for j in range(img_width):
			if(img[i][j] == 1):
				edge_point = 255
			elif(img[i][j] == 0):
				for k in range(3):
					for l in range(3):
						if(i + k - 1 >= 0 and i + k - 1 < img_height and j + l - 1 >= 0 and j + l - 1 < img_width):
							if(img[i + k - 1][j + l - 1] == 1):
								edge_point = 255
								break
					if(edge_point == 255):
						break
			else:
				edge_point = 0
			img_label[i][j] = edge_point
	return img_label
# Canny edge detection on img1
img1_noise_filter = noise_reduction(img1)
img1_gradient, theta1 = compute_gradient(img1_noise_filter, 1)
img1_non_maximal = non_maximal_suppression(img1_gradient, theta1)
img1_thresholding = hysteretic_thresholding(img1_non_maximal, 30, 10)
img1_label = connected_component_labeling(img1_thresholding)
cv2.imwrite("result3.jpg", img1_label)
print("Problem 1 (a) (3) done.")
# edge crispening method
img1_edge_crispening = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
b4 = 1
H = numpy.array([[1, b4, 1] ,[b4, math.pow(b4, 2), b4] ,[1, b4, 1]]) / math.pow(b4 + 2, 2)
edge_crispening_filter = signal.convolve2d(img1, H, mode = "same")
c = 3 / 5
img1_edge_crispening = (c / (2 * c - 1)) * img1 - ((1 - c) / (2 * c - 1)) * edge_crispening_filter
cv2.imwrite("result4.jpg", img1_edge_crispening)
# edge map
img1_edge_crispening_noise_filter = noise_reduction(img1_edge_crispening)
img1_edge_crispening_gradient, theta_a4 = compute_gradient(img1_edge_crispening_noise_filter, 1)
img1_edge_crispening_non_maximal = non_maximal_suppression(img1_edge_crispening_gradient, theta_a4)
img1_edge_crispening_thresholding = hysteretic_thresholding(img1_edge_crispening_non_maximal, 30, 10)
img1_edge_crispening_label = connected_component_labeling(img1_edge_crispening_thresholding)
cv2.imwrite("result5.jpg", img1_edge_crispening_label)
print("Problem 1 (a) (4) done.")
# obtain the edge map of sample2.jpg
img2_noise_filter = noise_reduction(img2)
img2_gradient, theta2 = compute_gradient(img2_noise_filter, 1)
img2_non_maximal = non_maximal_suppression(img2_gradient, theta2)
img2_thresholding = hysteretic_thresholding(img2_non_maximal, 10, 5)
img2_label = connected_component_labeling(img2_thresholding)
cv2.imwrite("sample2_edge_map.jpg", img2_label)
print("Problem 1 (b) done.")
print("Problem 1 done.\n")

# Problem 2: GEOMETRICAL MODIFICATION
img3 = cv2.imread("sample3.jpg", cv2.IMREAD_GRAYSCALE)
img3_height = img3.shape[0]
img3_width = img3.shape[1]
img5 = cv2.imread("sample5.jpg", cv2.IMREAD_GRAYSCALE)
img5_height = img5.shape[0]
img5_width = img5.shape[1]
# make sample3.jpg become sample4.jpg
img3_rotate = numpy.zeros((img3_height, img3_width), dtype = 'uint8')
for i in range(img3_height):
	for j in range(img3_width):
		img3_rotate[i][j] = img3[round(0.6 * j) + 200][img3_height - round(0.6 * i) + 50]
cv2.imwrite("result6.jpg", img3_rotate)
print("Problem 2 (a) done.")
# black hole
img5_max = max(img5_height, img5_width)
img5_black_hole = numpy.zeros((img5_max, img5_max), dtype = 'uint8')
row_shift = 1280 - 1024
col_shift = 0
for i in range(img5_height):
	for j in range(img5_width):
		img5_black_hole[i + row_shift][j + col_shift] = img5[i][j]
# rotate by middle point
def rotation(img):
	img_height = img.shape[0]
	img_width = img.shape[1]
	img_rotate = numpy.zeros((img_height, img_width), dtype = 'uint8')
	middle = img_height / 2
	for i in range(1, img_height - 1):
		for j in range(1, img_width - 1):
			angle = 0
			# rotation point
			if(i - middle == 0 and j - middle == 0):
				continue
			# rotate angle
			angle = ((numpy.hypot(middle, middle) - numpy.hypot(abs(i - middle), abs(j - middle))) / numpy.hypot(middle, middle)) * 180
			theta = numpy.radians(-angle)
			c = math.cos(theta)
			s = math.sin(theta)
			R = numpy.array(((c, -s), (s, c)))
			# coordinate
			coordinate = numpy.array([float(i - middle), float(j - middle)])
			rotate_point = R.dot(coordinate)
			if(int(int(rotate_point[0]) + middle) >= 1280 or int(int(rotate_point[0]) + middle) < 0 or int(int(rotate_point[1]) + middle) >= 1280 or int(int(rotate_point[1]) + middle) < 0):
				continue
			img_rotate[i][j] = img[int(int(rotate_point[0]) + middle)][int(int(rotate_point[1]) + middle)]
	return img_rotate
img5_black_hole = rotation(img5_black_hole)
# resize
img5_black_hole_resize = numpy.zeros((img5_height, img5_width), dtype = 'uint8')
for i in range(img5_height):
	for j in range(img5_width):
		img5_black_hole_resize[i][j] = img5_black_hole[i + row_shift][j + col_shift]
cv2.imwrite("result7.jpg", img5_black_hole_resize)
print("Problem 2 (b) done.")
print("Problem 2 done.\n")