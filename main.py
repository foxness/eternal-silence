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

    coords = []
    for i in range(BAR_COUNT):
        x = xs[i]
        height = heights[i]
        width = BAR_WIDTHS[i]

        x1 = x
        y1 = -(height + NUMBER_IMAGE_HEIGHT)
        x2 = x1 + width
        y2 = y1 + NUMBER_IMAGE_HEIGHT

        coords.append([x1, y1, x2, y2])
    
    number_images = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in coords]

    return number_images

def save_image(image, name):
    mpimg.imsave(name, image, cmap = 'gray')

def main():
    image = get_img()
    image = crop_initial(image)
    image = rgb2gray(image)
    image = darken(image)
    image = base_crop(image)
    image = crop_plot(image)
    number_images = get_bar_number_images(image)
    for i, img in enumerate(number_images):
        save_image(img, '{}.png'.format(i))
    # save_image(image)
    # show(image)

main()