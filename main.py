import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

CROP_INITIAL_TOP = 357
CROP_INITIAL_BOTTOM = 1765
PLOT_X = 45
BAR_WIDTHS = [137] * 3 + [138] + [137] * 3
BAR_GAP_WIDTH = 5
NUMBER_IMAGE_HEIGHT = 61
MAX_DIGIT_HEIGHT = 32
MAX_DIGIT_WIDTH = 22

WHITE = 1
BLACK = 0

BAR_COUNT = len(BAR_WIDTHS)

def get_img():
    return mpimg.imread('input.png')

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

def get_number_image_coords(xs, heights):
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
    
    return coords

def get_bar_number_images(image):
    xs = get_bar_xs()
    heights = get_bar_heights(image, xs)
    coords = get_number_image_coords(xs, heights)
    
    number_images = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in coords]

    return number_images

def save_image(image, name):
    mpimg.imsave(name, image, cmap = 'gray')

def horizontal_crop(image):
    img = image.copy()
    mask = (img == 0).any(axis = 1)

    return img[mask]

def vertical_crop(image):
    img = image.copy()

    mask = (img == 0).any(axis = 0)

    x1 = mask.argmax()
    x2 = len(mask) - mask[::-1].argmax()

    mask[x1:x2] = True

    return img.transpose()[mask].transpose()

def sequence_length(seq):
    current_elem = seq[0]
    current_length = 1

    sl = []
    for i, elem in enumerate(seq[1:]):
        if i == len(seq) - 2:
            sl.append(current_length + 1)

        if elem == current_elem:
            current_length += 1
        else:
            sl.append(current_length)
            current_length = 1
            current_elem = elem
    
    return sl

def sequence_accumulate(seq):
    s = np.array(seq)
    for i in range(len(s) - 1):
        s[(i + 1):] += seq[i]
    
    return s.tolist()

def digit_split(image):
    mask = (image == 0).any(axis = 0)
    sl = sequence_length(mask)
    sa = [0] + sequence_accumulate(sl)

    digit_count = int(len(sa) / 2)
    digit_coords = [[sa[i * 2], sa[i * 2 + 1]] for i in range(digit_count)]
    digit_images = [image[:, x1:x2] for (x1, x2) in digit_coords]

    return digit_images

def simplify(image):
    return darken(rgb2gray(image))

def load_digit_data():
    data = {}
    for path in glob.glob('digits\*.png'):
        image = expand_digit(simplify(mpimg.imread(path)))
        digit = path[-5:-4]
        data[digit] = image
    
    return data

def expand_digit(image):
    img = np.ones((MAX_DIGIT_HEIGHT, MAX_DIGIT_WIDTH))
    img[:image.shape[0], :image.shape[1]] = image
    return img

def difference(a, b):
    return np.absolute(a - b).sum()

def recognize_digit(image, data):
    diffs = [[digit, difference(image, digit_image)] for digit, digit_image in data.items()]
    return min(diffs, key = lambda diff: diff[1])[0]

def main():
    image = get_img()
    image = crop_initial(image)
    image = simplify(image)
    image = base_crop(image)
    image = crop_plot(image)

    images = get_bar_number_images(image)
    images = [horizontal_crop(img) for img in images]
    images = [vertical_crop(img) for img in images]
    images = [digit_split(img) for img in images]
    images = [[horizontal_crop(digit) for digit in digits] for digits in images]
    images = [[expand_digit(digit) for digit in digits] for digits in images]

    data = load_digit_data()

    print(recognize_digit(images[2][0], data))

    # for i, img in enumerate(images):
    #     for j, digit in enumerate(img):
    #         save_image(digit, '{} {} {}.png'.format(i, j, recognize_digit(digit, data)))

main()