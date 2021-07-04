import cv2
import numpy
import math
from scipy import signal

# Problem 1: DIGITAL HALFTONING
img1 = cv2.imread("sample1.png", cv2.IMREAD_GRAYSCALE)
img1_height = img1.shape[0]
img1_width = img1.shape[1]
# dithering using the dither matrix I2
img1_dithering_I2 = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
I2 = numpy.array([[1, 2], [3, 0]], dtype = 'uint8')
size = len(I2)
threshold = 255 * (I2 + 0.5) / (size * size)
ri = math.ceil(img1_height / size)
rj = math.ceil(img1_width / size)
img1_padding_I2 = numpy.zeros((ri * size, rj * size), dtype = 'uint8')
for i in range(img1_height):
	for j in range(img1_width):
		img1_padding_I2[i][j] = img1[i][j]
threshold = numpy.tile(threshold, (ri, rj))
img1_dithering_I2_temp = (img1_padding_I2 >= threshold)
for i in range(img1_height):
	for j in range(img1_width):
		img1_dithering_I2[i][j] = img1_dithering_I2_temp[i][j] * 255
cv2.imwrite("result1.png", img1_dithering_I2)
print("Problem 1 (a) done.")
# dithering using the dither matrix I256
img1_dithering_I256 = numpy.zeros((img1_height, img1_width), dtype = 'uint8')
I = I2
for i in range(7):
	temp = numpy.zeros((2 * I.shape[0], 2 * I.shape[1]), dtype = 'uint32')
	temp[0:I.shape[0], 0:I.shape[1]] = 4 * I + 1
	temp[0:I.shape[0], I.shape[1]:2 * I.shape[1]] = 4 * I + 2
	temp[I.shape[0]:2 * I.shape[0], 0:I.shape[1]] = 4 * I + 3
	temp[I.shape[0]:2 * I.shape[0], I.shape[1]:2 * I.shape[1]] = 4 * I + 0
	I = temp
I256 = I
size = len(I256)
threshold = 255 * (I256 + 0.5) / (size * size)
ri = math.ceil(img1_height / size)
rj = math.ceil(img1_width / size)
img1_padding_I256 = numpy.zeros((ri * size, rj * size), dtype = 'uint8')
for i in range(img1_height):
	for j in range(img1_width):
		img1_padding_I256[i][j] = img1[i][j]
threshold = numpy.tile(threshold, (ri, rj))
img1_dithering_I256_temp = (img1_padding_I256 >= threshold)
for i in range(img1_height):
	for j in range(img1_width):
		img1_dithering_I256[i][j] = img1_dithering_I256_temp[i][j] * 255
cv2.imwrite("result2.png", img1_dithering_I256)
print("Problem 1 (b) done.")
# error diffusion with two different filter masks
# Floyd Steinberg: https://stackoverflow.com/questions/55874902/floyd-steinberg-implementation-python
img1_Floyd_Steinberg = img1
for i in range(img1_height):
	for j in range(img1_width):
		#threshold
		old_value = img1_Floyd_Steinberg[i][j]
		if(old_value >= 128):
			new_value = 255
		else:
			new_value = 0
		img1_Floyd_Steinberg[i][j] = new_value
		# error
		error = old_value - new_value
		# Case 1: right
		if(j < img1_width - 1):
			new_number = img1_Floyd_Steinberg[i][j + 1] + error * 7 / 16
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Floyd_Steinberg[i][j + 1] = new_number
		# Case 2: lower left
		if(i < img1_height - 1 and j > 0):
			new_number = img1_Floyd_Steinberg[i + 1][j - 1] + error * 3 / 16
			if(new_number > 255):
				new_number = 255
			elif (new_number < 0):
				new_number = 0
			img1_Floyd_Steinberg[i + 1][j - 1] = new_number
		# Case 3: lower middle
		if(i < img1_height - 1):
			new_number = img1_Floyd_Steinberg[i + 1][j] + error * 5 / 16
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Floyd_Steinberg[i + 1][j] = new_number
		# Case 4: lower right
		if(i < img1_height - 1 and j < img1_width - 1):
			new_number = img1_Floyd_Steinberg[i + 1][j + 1] + error * 1 / 16
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Floyd_Steinberg[i + 1][j + 1] = new_number
cv2.imwrite("result3.png", img1_Floyd_Steinberg)
# Jarvis et al
img1_Jarvis = img1
for i in range(img1_height):
	for j in range(img1_width):
		#threshold
		old_value = img1_Jarvis[i][j]
		if(old_value >= 128):
			new_value = 255
		else:
			new_value = 0
		img1_Jarvis[i][j] = new_value
		# error
		error = old_value - new_value
		# Case 1:
		if(j < img1_width - 1):
			new_number = img1_Jarvis[i][j + 1] + error * 7 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i][j + 1] = new_number
		# Case 2:
		if(j < img1_width - 2):
			new_number = img1_Jarvis[i][j + 2] + error * 5 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i][j + 2] = new_number
		# Case 3:
		if(i < img1_height - 1 and j > 0):
			new_number = img1_Jarvis[i + 1][j - 2] + error * 3 / 48
			if(new_number > 255):
				new_number = 255
			elif (new_number < 0):
				new_number = 0
			img1_Jarvis[i + 1][j - 2] = new_number
		# Case 4:
		if(i < img1_height - 1 and j > 1):
			new_number = img1_Jarvis[i + 1][j - 1] + error * 5 / 48
			if(new_number > 255):
				new_number = 255
			elif (new_number < 0):
				new_number = 0
			img1_Jarvis[i + 1][j - 1] = new_number
		# Case 5:
		if(i < img1_height - 1):
			new_number = img1_Jarvis[i + 1][j] + error * 7 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 1][j] = new_number
		# Case 6:
		if(i < img1_height - 1 and j < img1_width - 1):
			new_number = img1_Jarvis[i + 1][j + 1] + error * 5 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 1][j + 1] = new_number
		# Case 7:
		if(i < img1_height - 1 and j < img1_width - 2):
			new_number = img1_Jarvis[i + 1][j + 2] + error * 3 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 1][j + 2] = new_number
		# Case 8:
		if(i < img1_height - 2 and j > 0):
			new_number = img1_Jarvis[i + 2][j - 2] + error * 1 / 48
			if(new_number > 255):
				new_number = 255
			elif (new_number < 0):
				new_number = 0
			img1_Jarvis[i + 2][j - 2] = new_number
		# Case 9:
		if(i < img1_height - 2 and j > 1):
			new_number = img1_Jarvis[i + 2][j - 1] + error * 3 / 48
			if(new_number > 255):
				new_number = 255
			elif (new_number < 0):
				new_number = 0
			img1_Jarvis[i + 2][j - 1] = new_number
		# Case 10:
		if(i < img1_height - 2):
			new_number = img1_Jarvis[i + 2][j] + error * 5 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 2][j] = new_number
		# Case 11:
		if(i < img1_height - 2 and j < img1_width - 1):
			new_number = img1_Jarvis[i + 2][j + 1] + error * 3 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 2][j + 1] = new_number
		# Case 12:
		if(i < img1_height - 2 and j < img1_width - 2):
			new_number = img1_Jarvis[i + 2][j + 2] + error * 1 / 48
			if(new_number > 255):
				new_number = 255
			elif(new_number < 0):
				new_number = 0
			img1_Jarvis[i + 2][j + 2] = new_number
cv2.imwrite("result4.png", img1_Jarvis)
print("Problem 1 (c) done.")
# transfer result1.png to a dotted halftone / manga style binary image
img1_dithering_I2_dotted = img1_dithering_I2
for i in range(4, img1_height, 256 // 32):
	for j in range(4, img1_width, 256 // 32):
		count = 0
		for k in range(i - 4, i + 4):
			for l in range(j - 4, j + 4):
				if(img1_dithering_I2_dotted[k][l] == 255):
					count += 1
		cv2.circle(img1_dithering_I2_dotted, (i, j), round(math.sqrt(count) // 2), (255, 255, 255), -1)
cv2.imwrite("result1_dotted.png", img1_dithering_I2_dotted)
print("Problem 1 (d) done.")
print("Problem 1 done.\n")

# Problem 2: FREQUENCY DOMAIN
img2 = cv2.imread("sample2.png", cv2.IMREAD_GRAYSCALE)
img2_height = img2.shape[0]
img2_width = img2.shape[1]
img3 = cv2.imread("sample3.png", cv2.IMREAD_GRAYSCALE)
img3_height = img3.shape[0]
img3_width = img3.shape[1]
# Fourier transform on sample2.png
f = numpy.fft.fft2(img2)
fshift = numpy.fft.fftshift(f)
magnitude_spectrum = 20 * numpy.log(numpy.abs(fshift))
cv2.imwrite("result5.png", magnitude_spectrum)
print("Problem 2 (a) done.")
# inverse Fourier transform: low-pass filter
fshift[0:200, 0:img2_width] = 0
fshift[0:img2_height, 0:200] = 0
fshift[0:img2_height, img2_width - 200:img2_width] = 0
fshift[img2_height - 200:img2_height, 0:img2_width] = 0
f_ishift = numpy.fft.ifftshift(fshift)
img2_low_pass_inverse_Fourier = numpy.fft.ifft2(f_ishift)
img2_low_pass_inverse_Fourier = numpy.abs(img2_low_pass_inverse_Fourier)
cv2.imwrite("result6.png", img2_low_pass_inverse_Fourier)
# low-pass filter in pixel domain: Gaussian
b = 2
H = (numpy.array([[1, b, 1] ,[b, math.pow(b, 2), b] ,[1, b, 1]])) / math.pow(b + 2, 2)
img2_low_pass = signal.convolve2d(img2, H, mode = "same")
cv2.imwrite("result7.jpg", img2_low_pass)
print("Problem 2 (b) done.")
# inverse Fourier transform: high-pass filter
center_row =  img2_height // 2
center_col =  img2_width // 2
fshift[center_row - 5:center_row + 5, center_col - 5:center_col + 5] = 0
f_ishift = numpy.fft.ifftshift(fshift)
img2_high_pass_inverse_Fourier = numpy.fft.ifft2(f_ishift)
img2_high_pass_inverse_Fourier = numpy.abs(img2_high_pass_inverse_Fourier)
cv2.imwrite("result8.png", img2_high_pass_inverse_Fourier)
# high-pass filter in pixel domain: Laplacian
H = numpy.array([[0, 1, 0] ,[1, -4, 1] ,[0, 1, 0]])
img2_high_pass = signal.convolve2d(img2, H, mode = "same")
cv2.imwrite("result9.jpg", img2_high_pass)
print("Problem 2 (c) done.")
# Fourier Transform on sample3.png
f = numpy.fft.fft2(img3)
fshift = numpy.fft.fftshift(f)
magnitude_spectrum = 20 * numpy.log(numpy.abs(fshift))
cv2.imwrite("result10.png", magnitude_spectrum)
print("Problem 2 (d) done.")
# remove the undesired pattern
fshift[0:190, 0:img3_width] = 0
fshift[0:img3_height, 0:190] = 0
fshift[0:img3_height, img3_width - 190:img3_width] = 0
fshift[img3_height - 190:img3_height, 0:img3_width] = 0
f_ishift = numpy.fft.ifftshift(fshift)
img3_remove = numpy.fft.ifft2(f_ishift)
img3_remove = numpy.abs(img3_remove)
cv2.imwrite("result11.png", img3_remove)
print("Problem 2 (e) done.")
print("Problem 2 done.\n")