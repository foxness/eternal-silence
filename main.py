import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_Y_TOP = 357
CROP_Y_BOTTOM = 1765

def get_img():
    return mpimg.imread('asd.png')

def darken(image):
    img = image.copy()
    mask = img < 0.999
    img[mask] = 0
    img[np.logical_not(mask)] = 1
    return img

def crop(image):
    return image[CROP_Y_TOP:CROP_Y_BOTTOM, :]

def show(image):
    plt.imshow(image, cmap = 'gray')
    plt.show()

def rgb2gray(image): #[0.2989, 0.5870, 0.1140] [0.1140, 0.5870, 0.2989]
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def mark_plot(image):
    for y in range(image.shape[0]):
        if image[y, 44] == 1 and image[y, 45] == 0:
            print('({}, {})'.format(45, y))

def main():
    image = get_img()
    image = crop(image)
    image = rgb2gray(image)
    image = darken(image)
    mark_plot(image)
    show(image)

main()