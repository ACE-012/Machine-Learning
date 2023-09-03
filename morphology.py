import cv2
import numpy as np
# import matplotlib as plt
while(1):
	image =cv2.imread("greyscale.png", cv2.IMREAD_COLOR)
	
	# Converts to HSV color space, OCV reads colors as BGR
	# frame is converted to hsv
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# print(hsv)
	# defining the range of masking
	blue1 = np.array([0, 0, 60])
	blue2 = np.array([0, 0, 255])
	
	# initializing the mask to be
	# convoluted over input image
	mask = cv2.inRange(hsv, blue1, blue2)
	# plt.plot(hsv, color="red")

	# plt.show()

	# passing the bitwise_and over
	# each pixel convoluted
	# res = cv2.bitwise_and(image, image, mask = mask)
	
	# cv2.imshow('res', res)
	# defining the kernel i.e. Structuring element
	kernel = np.ones((5, 5), np.uint8)
	
	# defining the opening function
	# over the image and structuring element
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	# The mask and opening operation
	# is shown in the window
	cv2.imshow('Original', image)
	cv2.imshow('Mask', mask)
	cv2.imshow('Opening', opening)
	
	# Wait for 'a' key to stop the program
	if cv2.waitKey(1) & 0xFF == ord('a'):
		break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
