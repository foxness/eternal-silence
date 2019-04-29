import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_Y_TOP = 357
CROP_Y_BOTTOM = 1765

def get_img():
    return mpimg.imread('asd.png')

def darken(image):
    img = image.copy()
    mask = img != 1
    img[mask] = 0
    return img

def crop(image):
    return image[CROP_Y_TOP:CROP_Y_BOTTOM, :]

def show(image):
    plt.imshow(image)
    plt.show()

def main():
    image = get_img()
    image = darken(image)
    image = crop(image)
    show(image)

main()