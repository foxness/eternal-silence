import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_Y_TOP = 357
CROP_Y_BOTTOM = 1765
PLOT_X = 45
WHITE = 1
BLACK = 0

def get_img():
    return mpimg.imread('asd.png')

def darken(image):
    img = image.copy()
    mask = img < 0.999
    img[mask] = BLACK
    img[np.logical_not(mask)] = WHITE
    return img

def crop(image):
    return image[CROP_Y_TOP:CROP_Y_BOTTOM, :]

def show(image):
    plt.imshow(image, cmap = 'gray')
    plt.show()

def rgb2gray(image): #[0.2989, 0.5870, 0.1140] [0.1140, 0.5870, 0.2989]
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def base_crop(image):
    for y in range(image.shape[0]):
        if image[y, PLOT_X - 1] == WHITE and image[y, PLOT_X] == BLACK:
            base = y
            break
    
    while image[base + 1, PLOT_X] == BLACK:
        base += 1
    
    return image[:base, :]

def main():
    image = get_img()
    image = crop(image)
    image = rgb2gray(image)
    image = darken(image)
    image = base_crop(image)
    show(image)

main()