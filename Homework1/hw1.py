import cv2
import numpy
import matplotlib.pyplot as plt
import math

# Problem 1: WARM-UP
img1 = cv2.imread("sample1.jpg")
img1_height = img1.shape[0]
img1_width = img1.shape[1]
# grayscale
img1_gray = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
for i in range(img1_height):
	for j in range(img1_width):
		img1_gray[i][j] = img1[i][j][0] * 0.299 + img1[i][j][1] * 0.587 + img1[i][j][2] * 0.114
cv2.imwrite("1_result.jpg", img1_gray)
print("Problem 1 (a) done.")
# horizontal flipping
img1_flip = numpy.zeros((img1_height, img1_width, 3), dtype = 'uint8')
for i in range(img1_width):
	img1_flip[:, i, :] = img1[:, img1_width - 1 - i, :]
cv2.imwrite("2_result.jpg", img1_flip)
print("Problem 1 (b) done.")
print("Problem 1 done.\n")

# Problem 2: IMAGE ENHANCEMENT
img2 = cv2.imread("sample2.jpg", cv2.IMREAD_GRAYSCALE)
img2_height = img2.shape[0]
img2_width = img2.shape[1]
img3 = cv2.imread("sample3.jpg", cv2.IMREAD_GRAYSCALE)
img3_height = img3.shape[0]
img3_width = img3.shape[1]
img4 = cv2.imread("sample4.jpg", cv2.IMREAD_GRAYSCALE)
img4_height = img4.shape[0]
img4_width = img4.shape[1]
# decrease the brightness of sample2.jpg
img2_decrease = numpy.zeros((img2_height, img2_width), dtype = 'uint8')
for i in range(img2_height):
	for j in range(img2_width):
		img2_decrease[i][j] = img2[i][j] / 5
cv2.imwrite("3_result.jpg", img2_decrease)
print("Problem 2 (a) done.")
# increase the brightness of 3_result.jpg
img2_increase = numpy.zeros((img2_height, img2_width), dtype = 'uint8')
for i in range(img2_height):
	for j in range(img2_width):
		img2_increase[i][j] = img2_decrease[i][j] * 5
cv2.imwrite("4_result.jpg", img2_increase)
print("Problem 2 (b) done.")
# plot the histograms of sample2.jpg, 3_result.jpg and 4_result.jpg
# sample2.jpg
img2_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img2_height):
	for j in range(img2_width):
		img2_hist[img2[i][j]] += 1
plt.bar(range(256), img2_hist, width = 1)
plt.savefig("sample2_hist.jpg")
plt.close()
# 3_result.jpg
img2_decrease_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img2_height):
	for j in range(img2_width):
		img2_decrease_hist[img2_decrease[i][j]] += 1
plt.bar(range(256), img2_decrease_hist, width = 1)
plt.savefig("3_result_hist.jpg")
plt.close()
# 4_result.jpg
img2_increase_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img2_height):
	for j in range(img2_width):
		img2_increase_hist[img2_increase[i][j]] += 1
plt.bar(range(256), img2_increase_hist, width = 1)
plt.savefig("4_result_hist.jpg")
plt.close()
print("Problem 2 (c) done.")
# global histogram equalization on sample3.jpg
img3_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img3_height):
	for j in range(img3_width):
		img3_hist[img3[i][j]] += 1
img3_CDF = numpy.zeros(256, dtype = 'uint64')
sum = 0
for i in range(256):
	temp = (img3_hist[i] * 255) / (img3_height * img3_width)
	sum += temp
	img3_CDF[i] = sum
img3_global = numpy.zeros((img3_height, img3_width), dtype = 'uint8')
for i in range(img3_height):
	for j in range(img3_width):
		img3_global[i][j] = img3_CDF[img3[i][j]]
cv2.imwrite("5_result.jpg", img3_global)
print("Problem 2 (d) done.")
# local histogram equalization on sample3.jpg
img3_local = numpy.zeros((img3_height, img3_width), dtype = 'uint8')
# boundary extension (even)
def boundary_extension(img, frame_size = 3):
	img_height = img.shape[0]
	img_width = img.shape[1]
	new_img = numpy.zeros((img_height + 2, img_width + 2), dtype = 'uint8')
	# corner
	new_img[0][0] = img[0][0]
	new_img[img_height + 1][0] = img[img_height - 1][0]
	new_img[0][img_width + 1] = img[0][img_width - 1]
	new_img[img_height + 1][img_width + 1] = img[img_height - 1][img_width - 1]
	# edge
	for i in range(img_height):
		new_img[i + 1][0] = img[i][0]
		new_img[i + 1][img_width + 1] = img[i][img_width - 1]
	for i in range(img_width):
		new_img[0][i + 1] = img[0][i]
		new_img[img_height + 1][i + 1] = img[img_height - 1][i]
	# center
	for i in range(img_height):
		for j in range(img_width):
			new_img[i + 1][j + 1] = img[i][j]
	if(frame_size == 3):
		return new_img
	else:
		return boundary_extension(new_img, frame_size - 2)
# window size = 51
window_length = 51
window_size = window_length * window_length
img3_extension = boundary_extension(img3, window_length)
for i in range(window_length // 2, img3_height + window_length // 2):
	for j in range(window_length // 2, img3_width + window_length // 2):
		count = window_size
		for k in range(window_length):
			for l in range(window_length):
				if(img3_extension[i + k - window_length // 2][j + l - window_length // 2] > img3_extension[i][j]):
					count -= 1
		img3_local[i - window_length // 2][j - window_length // 2] = count * 255 / window_size
cv2.imwrite("6_result.jpg", img3_local)
print("Problem 2 (e) done.")
# plot the histograms of 5_result.jpg and 6_result.jpg
# 5_result.jpg
img3_global_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img3_height):
	for j in range(img3_width):
		img3_global_hist[img3_global[i][j]] += 1
plt.bar(range(256), img3_global_hist, width = 1)
plt.savefig("5_result_hist.jpg")
plt.close()
# 6_result.jpg
img3_local_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img3_height):
	for j in range(img3_width):
		img3_local_hist[img3_local[i][j]] += 1
plt.bar(range(256), img3_local_hist, width = 1)
plt.savefig("6_result_hist.jpg")
plt.close()
print("Problem 2 (f) done.")
# enhance sample4.jpg
# transfer function: power-law transform
img4_power_law = numpy.zeros((img4_height, img4_width), dtype = 'uint8')
for i in range(img4_height):
	for j in range(img4_width):
		img4_power_law[i][j] = math.pow((img4[i][j] / 255), 1 / 2) * 255
cv2.imwrite("7_result.jpg", img4_power_law)
# histograms
img4_power_law_hist = numpy.zeros(256, dtype = 'uint64')
for i in range(img4_height):
	for j in range(img4_width):
		img4_power_law_hist[img4_power_law[i][j]] += 1
plt.bar(range(256), img4_power_law_hist, width = 1)
plt.savefig("7_result_hist.jpg")
plt.close()
print("Problem 2 (g) done.")
print("Problem 2 done.\n")

# Problem 3: NOISE REMOVAL
img5 = cv2.imread("sample5.jpg", cv2.IMREAD_GRAYSCALE)
img5_height = img5.shape[0]
img5_width = img5.shape[1]
img6 = cv2.imread("sample6.jpg", cv2.IMREAD_GRAYSCALE)
img6_height = img6.shape[0]
img6_width = img6.shape[1]
img7 = cv2.imread("sample7.jpg", cv2.IMREAD_GRAYSCALE)
img7_height = img7.shape[0]
img7_width = img7.shape[1]
# proper filters to remove noise from sample6.jpg and sample7.jpg
# mask size = 3 * 3
mask_length = 3
mask_size = mask_length * mask_length
# sample6.jpg: low-pass filter
b = 2
H = (numpy.array([[1, b, 1] ,[b, math.pow(b, 2), b] ,[1, b, 1]])) / math.pow(b + 2, 2)
img6_remove = numpy.zeros((img6_height, img6_width), dtype = 'uint8')
img6_extension = boundary_extension(img6, mask_length)
for i in range(mask_length // 2, img6_height + mask_length // 2):
	for j in range(mask_length // 2, img6_width + mask_length // 2):
		pixel_sum = 0
		mask_sum = 0
		for k in range(mask_length):
			for l in range(mask_length):
				pixel_sum += img6_extension[i + k - mask_length // 2][j + l - mask_length // 2] * H[k][l]
				mask_sum += H[k][l]
		img6_remove[i - mask_length // 2][j - mask_length // 2] = pixel_sum / mask_sum
cv2.imwrite("8_result.jpg", img6_remove)
# sample7.jpg: median filter
filt_times = 2
img7_remove = numpy.zeros((img7_height, img7_width), dtype = 'uint8')
mask_list = list()
for times in range(filt_times):
	img7_extension = boundary_extension(img7, mask_length)
	for i in range(mask_length // 2, img7_height + mask_length // 2):
		for j in range(mask_length // 2, img7_width + mask_length // 2):
			mask_list.clear()
			for k in range(-mask_length // 2, mask_length // 2 + 1):
				for l in range(-mask_length // 2, mask_length // 2 + 1):
					mask_list.append(img7_extension[i + k][j + l])
			img7_remove[i - mask_length // 2][j - mask_length // 2] = numpy.median(mask_list)
	img7 = img7_remove
cv2.imwrite("9_result.jpg", img7_remove)
print("Problem 3 (a) done.")
# PSNR values of 8_result.jpg and 9_result.jpg
# 8_result.jpg
MSE_8_result = 0
for i in range(img5_height):
	for j in range(img5_width):
		MSE_8_result += math.pow(int(img5[i][j]) - int(img6_remove[i][j]), 2)
MSE_8_result /= (img5_height * img5_width)
PSNR_8_result = 10 * math.log(math.pow(255, 2) / MSE_8_result, 10)
print("PSNR values of 8_result.jpg = %f" %(PSNR_8_result))
# 9_result.jpg
MSE_9_result = 0
for i in range(img5_height):
	for j in range(img5_width):
		MSE_9_result += math.pow(int(img5[i][j]) - int(img7_remove[i][j]), 2)
MSE_9_result /= (img5_height * img5_width)
PSNR_9_result = 10 * math.log(math.pow(255, 2) / MSE_9_result, 10)
print("PSNR values of 9_result.jpg = %f" %(PSNR_9_result))
print("Problem 3 (b) done.")
print("Problem 3 done.\n")