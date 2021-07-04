import cv2
import numpy
from scipy import signal
import pandas
from PIL import Image
import matplotlib.pyplot as plt
import math

# Problem 1: MORPHOLOGICAL PROCESSING
img1 = cv2.imread("sample1.png", cv2.IMREAD_GRAYSCALE)
img1_height = img1.shape[0]
img1_width = img1.shape[1]
# boundary extraction
img1_boundary = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
H = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = 'uint8')
if(numpy.max(img1) == 255):
	F = numpy.where(img1 == 255, 1, 0)
else:
	F = img1
# generalized dilation and erosion
def general_dilation_erosion(img, H, method):
	if(numpy.max(img) == 255):
		F = numpy.where(img == 255, 1, 0)
	else:
		F = img
	T = signal.convolve2d(F, H, mode = "same")
	if(method == "erosion"):
		G = numpy.where(T == numpy.sum(H), 1, 0)
	elif(method == "dilation"):
		G = numpy.where(T > 0, 1, 0)
	return G
img1_boundary = (F - general_dilation_erosion(F, H, "erosion")) * 255
cv2.imwrite("result1.png", img1_boundary)
print("Problem 1 (a) done.")
# hole filling
img1_hole = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
H = numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = 'uint8')
img1_comp = 1 - F
def hole_fill(img, Fc, H, x, y):
	if(numpy.max(img) == 255):
		F = numpy.where(img == 255, 1, 0)
	else:
		F = img
	Gi = numpy.zeros((img.shape[0], img.shape[1]))
	G = numpy.zeros((img.shape[0], img.shape[1]))
	Gi[x][y] = 1
	flag = 0
	while(flag == 0):
		G = general_dilation_erosion(Gi, H, "dilation") * Fc
		flag = 1
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if(G[i][j] != Gi[i][j]):
					flag = 0
		Gi = G
	G = (Gi + F) * 255
	return G
img1_hole = hole_fill(img1, img1_comp, H, 189, 49)
img1_hole = hole_fill(img1_hole, img1_comp, H, 169, 84)
img1_hole = hole_fill(img1_hole, img1_comp, H, 304, 163)
img1_hole = hole_fill(img1_hole, img1_comp, H, 195, 298)
img1_hole = hole_fill(img1_hole, img1_comp, H, 59, 122)
img1_hole = hole_fill(img1_hole, img1_comp, H, 232, 63)
img1_hole = hole_fill(img1_hole, img1_comp, H, 93, 264)
img1_hole = hole_fill(img1_hole, img1_comp, H, 93, 128)
cv2.imwrite("result2.png", img1_hole)
print("Problem 1 (b) done.")
# count the number of objects
H = numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = 'uint8')
def count_objects(img, H, x, y):
	if(numpy.max(img) == 255):
		F = numpy.where(img == 255, 1, 0)
	else:
		F = img
	Gi = numpy.zeros((img.shape[0], img.shape[1]))
	G = numpy.zeros((img.shape[0], img.shape[1]))
	Gi[x][y] = 1
	flag = 0
	while(flag == 0):
		G = general_dilation_erosion(Gi, H, "dilation") * F
		flag = 1
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if(G[i][j] != Gi[i][j]):
					flag = 0
		Gi = G
	G = (F - Gi) * 255
	return G
count = 0
for i in range(img1_height):
	for j in range(img1_width):
		if(img1_hole[i][j] / 255 == 1):
			count += 1
			img1_hole = count_objects(img1_hole, H, i, j)
print("the number of objects = %d" %(count))
print("Problem 1 (c) done.")
print("Problem 1 done.\n")

# Problem 2: TEXTURE ANALYSIS
img2 = cv2.imread("sample2.png", cv2.IMREAD_GRAYSCALE)
img2_height = img2.shape[0]
img2_width = img2.shape[1]
img3 = cv2.imread("sample3.png", cv2.IMREAD_GRAYSCALE)
img3_height = img3.shape[0]
img3_width = img3.shape[1]
# Law's method
window_size = 13
F = img2
# convolution
H1 = numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36
H2 = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12
H3 = numpy.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12
H4 = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12
H5 = numpy.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
H6 = numpy.array([[-1, 2,-1], [0, 0, 0], [1, -2, 1]]) / 4
H7 = numpy.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
H8 = numpy.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4
H9 = numpy.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
# micro-structure impulse response arrays
M1 = signal.convolve2d(F, H1, mode = "same")
M2 = signal.convolve2d(F, H2, mode = "same")
M3 = signal.convolve2d(F, H3, mode = "same")
M4 = signal.convolve2d(F, H4, mode = "same")
M5 = signal.convolve2d(F, H5, mode = "same")
M6 = signal.convolve2d(F, H6, mode = "same")
M7 = signal.convolve2d(F, H7, mode = "same")
M8 = signal.convolve2d(F, H8, mode = "same")
M9 = signal.convolve2d(F, H9, mode = "same")
# energy computation
S = numpy.ones((window_size, window_size))
T1 = signal.convolve2d(M1 * M1, S, mode = "same")
T2 = signal.convolve2d(M2 * M2, S, mode = "same")
T3 = signal.convolve2d(M3 * M3, S, mode = "same")
T4 = signal.convolve2d(M4 * M4, S, mode = "same")
T5 = signal.convolve2d(M5 * M5, S, mode = "same")
T6 = signal.convolve2d(M6 * M6, S, mode = "same")
T7 = signal.convolve2d(M7 * M7, S, mode = "same")
T8 = signal.convolve2d(M8 * M8, S, mode = "same")
T9 = signal.convolve2d(M9 * M9, S, mode = "same")
cv2.imwrite("law1.png", T1 / numpy.max(T1) * 255)
cv2.imwrite("law2.png", T2 / numpy.max(T2) * 255)
cv2.imwrite("law3.png", T3 / numpy.max(T3) * 255)
cv2.imwrite("law4.png", T4 / numpy.max(T4) * 255)
cv2.imwrite("law5.png", T5 / numpy.max(T5) * 255)
cv2.imwrite("law6.png", T6 / numpy.max(T6) * 255)
cv2.imwrite("law7.png", T7 / numpy.max(T7) * 255)
cv2.imwrite("law8.png", T8 / numpy.max(T8) * 255)
cv2.imwrite("law9.png", T9 / numpy.max(T9) * 255)
local_feature = numpy.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])
law_method = numpy.moveaxis(local_feature, 0, -1)
print("Problem 2 (a) done.")
# k-means algorithm: https://gist.github.com/tvwerkhoven/4fdc9baad760240741a09292901d3abd
def kMeans(X, K, iterations, plot_progress = None):
	# choose centroids randomly
	centroids = X[numpy.random.choice(numpy.arange(len(X)), K)]
	for i in range(iterations):
		# cluster assignment
		C = numpy.array([numpy.argmin([numpy.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
		# check if there are fewer than K clusters.
		if(len(numpy.unique(C)) < K):
			centroids = X[numpy.random.choice(numpy.arange(len(X)), K)]
		else:
			centroids = [X[C == k].mean(axis = 0) for k in range(K)]
	return numpy.array(centroids), C
centroids, cluster_raw = kMeans(law_method.reshape(-1, 9), 5, 10)
cluster_new = cluster_raw.reshape(law_method.shape[0], law_method.shape[1])
img2_kmeans = numpy.array(Image.new("RGB", (cluster_new.shape[1], cluster_new.shape[0])))
# output color image
for i, cluster_index in enumerate(numpy.unique(cluster_new)):
	position = numpy.argwhere(cluster_new == cluster_index)
	img2_kmeans[position[:, 0], position[:, 1], :] = numpy.array((plt.cm.tab10(i)[0], plt.cm.tab10(i)[1], plt.cm.tab10(i)[2])) * 255
cv2.imwrite("result3.png", img2_kmeans)
print("Problem 2 (b) done.")
# improve the classification result
# power-law transform
img2_power_law = numpy.zeros((img2_height, img2_width), dtype = 'uint8')
for i in range(img2_height):
	for j in range(img2_width):
		img2_power_law[i][j] = math.pow((img2[i][j] / 255), 2) * 255
cv2.imwrite("sample2_power_law.png", img2_power_law)
# Law's method
window_size = 13
F = img2_power_law
# convolution
H1 = numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36
H2 = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12
H3 = numpy.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12
H4 = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12
H5 = numpy.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
H6 = numpy.array([[-1, 2,-1], [0, 0, 0], [1, -2, 1]]) / 4
H7 = numpy.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
H8 = numpy.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4
H9 = numpy.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
# micro-structure impulse response arrays
M1 = signal.convolve2d(F, H1, mode = "same")
M2 = signal.convolve2d(F, H2, mode = "same")
M3 = signal.convolve2d(F, H3, mode = "same")
M4 = signal.convolve2d(F, H4, mode = "same")
M5 = signal.convolve2d(F, H5, mode = "same")
M6 = signal.convolve2d(F, H6, mode = "same")
M7 = signal.convolve2d(F, H7, mode = "same")
M8 = signal.convolve2d(F, H8, mode = "same")
M9 = signal.convolve2d(F, H9, mode = "same")
# energy computation
S = numpy.ones((window_size, window_size))
T1 = signal.convolve2d(M1 * M1, S, mode = "same")
T2 = signal.convolve2d(M2 * M2, S, mode = "same")
T3 = signal.convolve2d(M3 * M3, S, mode = "same")
T4 = signal.convolve2d(M4 * M4, S, mode = "same")
T5 = signal.convolve2d(M5 * M5, S, mode = "same")
T6 = signal.convolve2d(M6 * M6, S, mode = "same")
T7 = signal.convolve2d(M7 * M7, S, mode = "same")
T8 = signal.convolve2d(M8 * M8, S, mode = "same")
T9 = signal.convolve2d(M9 * M9, S, mode = "same")
local_feature = numpy.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])
law_method = numpy.moveaxis(local_feature, 0, -1)
# k-means algorithm
centroids, cluster_raw = kMeans(law_method.reshape(-1, 9), 5, 10)
cluster_new = cluster_raw.reshape(law_method.shape[0], law_method.shape[1])
img2_power_law_kmeans = numpy.array(Image.new("RGB", (cluster_new.shape[1], cluster_new.shape[0])))
# output color image
for i, cluster_index in enumerate(numpy.unique(cluster_new)):
	position = numpy.argwhere(cluster_new == cluster_index)
	img2_power_law_kmeans[position[:, 0], position[:, 1], :] = numpy.array((plt.cm.tab10(i)[0], plt.cm.tab10(i)[1], plt.cm.tab10(i)[2])) * 255
cv2.imwrite("result4.png", img2_power_law_kmeans)
print("Problem 2 (c) done.")
print("Problem 2 done.\n")