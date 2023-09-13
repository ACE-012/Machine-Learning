import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# import matplotlib as plt
image_B =cv2.imread("./Datasets/lung/Bengin cases/Bengin case (1).jpg", cv2.IMREAD_COLOR)
image_M =cv2.imread("./Datasets/lung/Malignant cases/Malignant case (1).jpg", cv2.IMREAD_COLOR)
image_2 =cv2.imread("./Datasets/lung/Malignant cases/Malignant case (2).jpg", cv2.IMREAD_COLOR)
image_3 =cv2.imread("./Datasets/lung/Malignant cases/Malignant case (3).jpg", cv2.IMREAD_COLOR)
image_4 =cv2.imread("./Datasets/lung/Malignant cases/Malignant case (4).jpg", cv2.IMREAD_COLOR)
	# image=plt.imread('./Datasets/lung/Bengin cases/Bengin case (1).jpg')
B=image_B.reshape(-1,3)
M=image_M.reshape(-1,3)
kmeans=KMeans(n_clusters=2,n_init=10)
kmeans.fit(B)
segmented_image_B=kmeans.cluster_centers_[kmeans.labels_]
segmented_image_B=segmented_image_B.reshape(image_B.shape)

kmeans_M=KMeans(n_clusters=2,n_init=10)
kmeans_M.fit(M)
segmented_image_M=kmeans_M.cluster_centers_[kmeans_M.labels_]
segmented_image_M=segmented_image_M.reshape(image_M.shape)
sub=cv2.subtract(segmented_image_B,segmented_image_M)
while(1):
	cv2.imshow('image_B', image_B)
	cv2.imshow('image_M', image_M)
	cv2.imshow('image_4', image_4)
	cv2.imshow('image_2', image_2)
	cv2.imshow('image_3', image_3)
	# plt.waitforbuttonpress()
	# plt.imshow(image_M)
	# plt.waitforbuttonpress()
	cv2.imshow('segmented_img_B',segmented_image_B/255)
	cv2.imshow('segmented_img_M',segmented_image_M/255)
	# plt.waitforbuttonpress()
	# plt.imshow(segmented_image_M/255)
	# plt.waitforbuttonpress()
	# sub=np.asanyarray(sub)
	cv2.imshow('sub',sub)
	# print()
	# Converts to HSV color space, OCV reads colors as BGR
	# frame is converted to hsv
	# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# print(hsv)
	# defining the range of masking
# 	blue1 = np.array([0, 0, 60])
# 	blue2 = np.array([0, 0, 255])
	
# 	# initializing the mask to be
# 	# convoluted over input image
# 	mask = cv2.inRange(hsv, blue1, blue2)
# 	# plt.plot(hsv, color="red")

# 	# plt.show()

# 	# passing the bitwise_and over
# 	# each pixel convoluted
# 	# res = cv2.bitwise_and(image, image, mask = mask)
	
# 	# cv2.imshow('res', res)
# 	# defining the kernel i.e. Structuring element
# 	kernel = np.ones((5, 5), np.uint8)
	
# 	# defining the opening function
# 	# over the image and structuring element
# 	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
# 	# The mask and opening operation
# 	# is shown in the window
# 	cv2.imshow('Original', image)
# 	cv2.imshow('Mask', mask)
# 	cv2.imshow('Opening', opening)
	
# 	# Wait for 'a' key to stop the program
	if cv2.waitKey(1) & 0xFF == ord('a'):
		break

# # De-allocate any associated memory usage
# cv2.destroyAllWindows()
