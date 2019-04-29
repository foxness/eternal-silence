import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_INITIAL_TOP = 357
CROP_INITIAL_BOTTOM = 1765
PLOT_X = 45
BAR_WIDTHS = [137] * 3 + [138] + [137] * 3
BAR_GAP_WIDTH = 5
NUMBER_IMAGE_HEIGHT = 61

WHITE = 1
BLACK = 0

BAR_COUNT = len(BAR_WIDTHS)

def get_img():
    return mpimg.imread('asd.png')

def darken(image):
    img = image.copy()
    mask = img < 0.999
    img[mask] = BLACK
    img[np.logical_not(mask)] = WHITE
    return img

def crop_initial(image):
    return image[CROP_INITIAL_TOP:CROP_INITIAL_BOTTOM, :]

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
    
    return image[:base + 1, :]

def crop_plot(image):
    return image[:, PLOT_X:-PLOT_X]

def get_bar_xs():
    xs = []
    for i in range(BAR_COUNT):
        x = 0
        for j in range(i):
            x += BAR_WIDTHS[j] + BAR_GAP_WIDTH
        
        xs.append(x)
    
    return xs

def get_bar_heights(image, bar_xs):
    heights = []
    for x in bar_xs:
        y = 0
        while image[-y - 1, x] == BLACK:
            y += 1
        
        heights.append(y)
    
    return heights

def get_bar_number_images(image):
    xs = get_bar_xs()
    heights = get_bar_heights(image, xs)

    return image

def save_image(image):
    mpimg.imsave('output.png', image, cmap = 'gray')

def main():
    image = get_img()
    image = crop_initial(image)
    image = rgb2gray(image)
    image = darken(image)
    image = base_crop(image)
    image = crop_plot(image)
    get_bar_number_images(image)
    save_image(image)
    show(image)

main()