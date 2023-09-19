# import cv2
# import numpy as np
# import time
# creation=time.time()
# normal =cv2.imread("greyscale.png", cv2.IMREAD_COLOR)
# bengin =cv2.imread("Datasets/lung/Bengin cases/Bengin case (1).jpg", cv2.IMREAD_COLOR)
# malig="Datasets/lung/Malignant cases/Malignant case ("
# nant=").jpg"
# malignant =cv2.imread("Datasets/lung/Malignant cases/Malignant case (1).jpg", cv2.IMREAD_COLOR)
# A=[]
# def Dialation(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     blue1 = np.array([0, 0, 90])
#     blue2 = np.array([0, 0, 255])
#     mask = cv2.inRange(hsv, blue1, blue2)
# 	# passing the bitwise_and over
# 	# each pixel convoluted
# 	# res = cv2.bitwise_and(image, image, mask = mask)
# 	# cv2.imshow('res', res)
# 	# defining the kernel i.e. Structuring element
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     return opening
# # cv2.imshow('Original', Dialation(normal))
# # cv2.imshow('bengin', Dialation(bengin))
# # cv2.imshow('malignant', Dialation(malignant))

# def compare(img1,img2):
#    diff = cv2.subtract(img1, img2)
#    err = np.sum(diff**2)
#    return (0 if err==0 else 1) 
# start = time.time()
# for i in range(500):
#     try:
#         img=cv2.imread(malig+str(i+1)+nant, cv2.IMREAD_COLOR)
#         val=compare(normal,img)
#         A.append(val)
#     except:
#         pass
# end=time.time()
# # for i in A:
# #     print(i)
# end1=time.time()
# print( (end-start) * 10**3, "ms")
# print("END",(end1-creation) * 10**3, "ms")
# # print(compare(normal,normal))
# # print(compare(normal,bengin))
# # print(compare(normal,malignant))
# # print(compare(malignant,bengin))
# # while True:
# # 	if cv2.waitKey(1) & 0xFF == ord('a'):
# 		# break
# # cv2.destroyAllWindows()
# # 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))