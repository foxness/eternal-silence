import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CROP_INITIAL_TOP = 357
CROP_INITIAL_BOTTOM = 1765
PLOT_X = 45
BAR_WIDTH = 137
BAR_GAP_WIDTH = 5
BAR_COUNT = 7

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

def asd(image):
    heights = []
    for x in [(BAR_WIDTH + BAR_GAP_WIDTH) * i for i in range(BAR_COUNT)]:
        print(x)
        y = 0
        while image[-y - 1, x] == BLACK:
            y += 1
        
        heights.append(y)
    
    print(heights)

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
    asd(image)
    save_image(image)
    show(image)

main()