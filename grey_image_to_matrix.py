from PIL import Image
from numpy import asarray,savetxt

img = Image.open('./Datasets/lung/Normal cases/Normal case (1).jpg').convert('L')
numpydata = asarray(img)
img.save('greyscale.png')
with open("test.txt","+a") as w:
    print(numpydata)
    savetxt(w,numpydata)
# import numpy as np
# from matplotlib import pyplot as plt

# random_image = np.random.random([500, 500])

# plt.imshow(random_image, cmap='gray')
# plt.colorbar()
# from scipy.ndimage import zoom
# from PIL import Image
# import numpy as np

# srcImage = Image.open("./Datasets/lung_image_sets/lung_aca/lungaca1.jpeg")
# grayImage = srcImage.convert('L')
# array = np.array(grayImage)
# array = zoom(array, 310/174)

# np.savetxt("binarized.txt", array<128, fmt="%d")
# print("\n\n Output Stored to binarized.txt.......#")