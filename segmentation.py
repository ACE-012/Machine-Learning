# Importing Necessary Libraries
from skimage import data
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Setting the plot size to 15,15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package
coffee =Image.open('greyscale.png').convert('L')
plt.subplot(1, 2, 1)

# Displaying the sample image
plt.imshow(coffee)

# Converting RGB image to Monochrome
gray_coffee = rgb2gray(coffee)
plt.subplot(1, 2, 2)

# Displaying the sample image - Monochrome
# Format
plt.imshow(gray_coffee, cmap="gray")
# Importing Necessary Libraries
# Displaying the sample image - Monochrome Format
# from skimage import data
# from skimage import filters
# from skimage.color import rgb2gray
# import matplotlib.pyplot as plt

# # Sample Image of scikit-image package
# coffee = data.coffee()
# gray_coffee = rgb2gray(coffee)

# # Setting the plot size to 15,15
# plt.figure(figsize=(15, 15))

# for i in range(10):

# # Iterating different thresholds
#     binarized_gray = (gray_coffee > i*0.1)*1
#     plt.subplot(5,2,i+1)

# # Rounding of the threshold
# # value to 1 decimal point
#     plt.title("Threshold: >"+str(round(i*0.1,1)))

# # Displaying the binarized image
# # of various thresholds
#     plt.imshow(binarized_gray, cmap = 'gray')

# plt.tight_layout()
# print("END")