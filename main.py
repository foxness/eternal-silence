import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_Y_TOP = 357
CROP_Y_BOTTOM = 1775

image = mpimg.imread('asd.png')

mask = image != 1
image[mask] = 0

image = image[CROP_Y_TOP:CROP_Y_BOTTOM, :]

print(image.shape)

plt.imshow(image)
plt.show()