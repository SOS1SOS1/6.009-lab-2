#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


# VARIOUS FILTERS

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_filter(color_img):
        red_pixels = [rgb_val[0] for rgb_val in color_img['pixels']]
        green_pixels = [rgb_val[1] for rgb_val in color_img['pixels']]
        blue_pixels = [rgb_val[2] for rgb_val in color_img['pixels']]

        h = color_img['height']
        w = color_img['width']
        red_pixels = filt({'height': h, 'width': w, 'pixels': red_pixels})['pixels']
        green_pixels = filt({'height': h, 'width': w, 'pixels': green_pixels})['pixels']
        blue_pixels = filt({'height': h, 'width': w, 'pixels': blue_pixels})['pixels']

        filtered_color_image = {'height': h, 'width': w, 'pixels': [(red_pixels[i], green_pixels[i], blue_pixels[i]) for i in range(h*w)]}
        return filtered_color_image

    return color_filter

def make_blur_filter(n):
    """
    returns a blurred image filter function that uses a box blur of size n
    """
    def blurred_image_filter(img):
        return blurred(img, n)
    return blurred_image_filter

def make_sharpen_filter(n):
    """
    returns a sharpened image filter function that uses a box blur of size n
    """
    def sharpened_image_filter(img):
        return sharpened(img, n)
    return sharpened_image_filter

def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def apply_filter_cascade(img):
        for filt in filters:
            img = filt(img)
        return img
    return apply_filter_cascade

# my custom filter
def darken(img_color, amount):
    """
    Takes in a color image and return a new image that is a darkened by the amount (value from 0 to 1) version of the image
    """
    if amount < 0 or amount > 1:
        return None
    darkened_image = {
        'height': img_color['height'],
        'width': img_color['width'],
        'pixels': [(round(rgb_val[0]*amount), round(rgb_val[1]*amount), round(rgb_val[2]*amount)) for rgb_val in img_color['pixels']]
    }
    return darkened_image

# Helper Functions from Lab 1

def get_pixel(image, x, y):
    # handles out of bound coordinates
    if x < 0: 
        x = 0
    elif x >= image['height']:
        x = image['height'] - 1
    if y < 0: 
        y = 0
    elif y >= image['width']:
        y = image['width'] - 1
    # used helper function to get index of (x, y) pixel in 1D array
    return image['pixels'][get_pixel_index(image, x, y)]
    #return image['pixels'][x, y)

def get_pixel_index(image, x, y):
    """
    Calculates and returns the index of the (x, y) pixel in the 1D pixels array
    """
    return (image['width'] * x) + y

def set_pixel(image, x, y, c):
    # used helper function to get index of (x, y) pixel in 1D array
    image['pixels'][get_pixel_index(image, x, y)] = c
    #image['pixels'][x, y] = c

def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'], # fixed typo in width (widht -> width)
        #'widht': image['width'],
        'pixels': image['pixels'].copy() # initialized results with pixels from original image
        #'pixels': [],
    }
    for x in range(image['height']):
        for y in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            # moved set pixel call inside of the second for loop, so it runs for each pixel and fixed the order of params, x and y
            set_pixel(result, x, y, newcolor)
        #set_pixel(result, y, x, newcolor)
    return result

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for x in range(image['height']):
        for y in range(image['width']):
            value = round(get_pixel(image, x, y))
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            set_pixel(image, x, y, value)

def create_box_blur_kernel(n):
    """
    Takes in a value n and outputs a 2d array of size n x n whose values sum to 1
    """
    total_values = n * n
    k = [[1/total_values] * n] * n
    return k

def inverted(image):
    # changed 256-c to 255-c
    return apply_per_pixel(image, lambda c: 255-c)

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    blurred_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    box_blur = create_box_blur_kernel(n)

    # then compute the correlation of the input image with that kernel
    blurred_image = correlate(image, box_blur)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(blurred_image)
    return blurred_image

def sharpened(image, n):
    """
    Returns a new sharpened image (also called unsharp mask) by subtracting a 
    blurred version of the image from a scaled version of the original image

    This process does not mutate the input image; rather, it creates a separate 
    structure to represent the output.
    """
    # create a new image with the same height and width as the inputed image and double all of the pixel values
    sharpened_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [ (2 * i) for i in image['pixels'] ]
    }
    blurred_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }
    # create the box blur kernel
    box_blur = create_box_blur_kernel(n)
    # compute the correlation of the inputed image with the box blur
    blurred_image = correlate(image, box_blur)

    # loop over the pixels in the sharpened image and subtract the corresponding blurred image pixel
    for i in range(len(sharpened_image['pixels'])):
        sharpened_image['pixels'][i] -= blurred_image['pixels'][i]

    # make sure it is a valid image
    round_and_clip_image(sharpened_image)
    return sharpened_image

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    kernel - 2D float array
    """
    correlated_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }

    kernel_size = len(kernel)
    # loops over the coords in the image
    for x in range(image['height']):
        for y in range(image['width']):
            # for each coord it loops over the kernel and calculates a correlated sum
            correlated_sum = 0
            # initializes image_x to the x pos in the image that corresponds to row 0 in the kernel
            image_x = int(x - ((kernel_size - 1) / 2))
            for i in range(kernel_size):
                # initializes image_y to the y pos in the image that corresponds to col 0 in the kernel
                image_y = int(y - ((kernel_size - 1) / 2))
                for j in range(kernel_size):
                    # get_pixel handles if the coords are out of bounds
                    correlated_sum += get_pixel(image, image_x, image_y) * kernel[i][j]
                    image_y += 1
                image_x += 1
            # updates the pixels value to be the new sum
            set_pixel(correlated_image, x, y, correlated_sum)
    return correlated_image

def edges(image):
    """
    Implements the Sobel operator filter, which is useful for detecting edges in images
        - Performs 2 separate correlations (one with k_x and one with k_y)
        - the resulting image is a combination of the 2 correlated images, according to the formula below
            round([c1^2+c2^2]^(1/2))
    """
    k_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    k_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    # inputed image correlated with kernel x
    o_x = correlate(image, k_x)
    # inputed image correlated with kernel y
    o_y = correlate(image, k_y)
    # setup a new image object with the same height and width as the inputed image
    edge_detected_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
    }
    # for each coord, calculate the square root of the sum of both correlated images squared
    for x in range(image['height']):
        for y in range(image['width']):
            edge_detected_image['pixels'].append(round(math.sqrt(get_pixel(o_x, x, y)*get_pixel(o_x, x, y) + get_pixel(o_y, x, y)*get_pixel(o_y, x, y))))
    # make sure it is a valid image
    round_and_clip_image(edge_detected_image)
    return edge_detected_image

# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    for i in range(ncols):
        grey = greyscale_image_from_color_image(image)
        energy = compute_energy(grey)
        energy_map = cumulative_energy_map(energy)
        seam = minimum_energy_seam(energy_map)
        image = image_without_seam(image, seam)
    return image

# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    return {'height': image['height'], 'width': image['width'], 'pixels': [round(.299*rgb_val[0]+.587*rgb_val[1]+.114*rgb_val[2]) for rgb_val in image['pixels']]}

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)

# added a helper function to get min adjacent pixel
def min_adj_pixel_above(img, i):
    min_adj = img['pixels'][i-img['width']]
    min_i = i - img['width']
    if i % img['width'] != 0 and min_adj >= img['pixels'][i-img['width']-1]: # >= allows for it to take left pixel in the case of a tie
        # has a pixel above and to the left
        min_adj = img['pixels'][i-img['width']-1]
        min_i = (i - img['width'] - 1)
    if (i+1) % img['width'] != 0 and min_adj > img['pixels'][i-img['width']+1]:
        # has a pixel above and to the right
        min_adj = img['pixels'][i-img['width']+1]
        min_i = (i - img['width'] + 1)
    return min_adj, min_i

    # 0 1 2
    # 3 4 5
    # 6 7 8

def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    cumulative_energy = {
        'height': energy['height'],
        'width': energy['width'],
        'pixels': energy['pixels'].copy()
    }
    for i in range(len(energy['pixels'])):
        if i >= energy['width']: # not in top row
            cumulative_energy['pixels'][i] += min_adj_pixel_above(cumulative_energy, i)[0]
    return cumulative_energy

def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    # loop over pixels bottom row up
    #   for bottom row - pick smallest energy and save that index
    #   after that - pick smallest adjacent index from last index
    indices = []
    for i in reversed(range(cem['height'])): # bottom up 
        if len(indices) > 0:
            # after it picks the min energy index for the bottom, it picks the next minimum index above
            last_index = indices[-1] # gets last index in list
            _, min_i = min_adj_pixel_above(cem, last_index)
            indices.append(min_i)
        else:
            # initialize min energy pixel in last row to last pixel in the last row
            cur_min_energy = cem['pixels'][len(cem['pixels']) - cem['width']]
            cur_min_energy_i = len(cem['pixels']) - cem['width']
            # loops over bottom row of image from right to left
            for j in reversed(range(cem['width'])):
                # >= makes sure the left pixel wins in a tie of min energy level
                if cur_min_energy >= cem['pixels'][get_pixel_index(cem, i, j)]:
                    cur_min_energy_i = get_pixel_index(cem, i, j)
                    cur_min_energy = cem['pixels'][cur_min_energy_i]
            indices.append(cur_min_energy_i)
    return indices

def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    no_seam_image = {
        'height': image['height'],
        'width': image['width']-1,
        'pixels': []
    }
    for i in range(len(image['pixels'])):
        if not i in seam:
            no_seam_image['pixels'].append(image['pixels'][i])
    return no_seam_image

# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}

def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}

def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # Section 4.1 - Filter on Color Images
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # cat = load_color_image('test_images/cat.png')
    # save_color_image(color_inverted(cat), 'test_images/inverted_cat.png')

    # Section 4.2 - Other Kinds of Filters
    # blurred_9 = make_blur_filter(9)
    # color_blurred_9 = color_filter_from_greyscale_filter(blurred_9)
    # python = load_color_image('test_images/python.png')
    # save_color_image(color_blurred_9(python), 'test_images/blurred_python.png')

    # Section 4.2 - Other Kinds of Filters
    # sharpened_7 = make_sharpen_filter(7)
    # color_sharpened_7 = color_filter_from_greyscale_filter(sharpened_7)
    # sparrowchick = load_color_image('test_images/sparrowchick.png')
    # save_color_image(color_sharpened_7(sparrowchick), 'test_images/sharpended_sparrowchick.png')

    # Section 5 - Cascade of Filters
    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # frog = load_color_image('test_images/frog.png')
    # save_color_image(filt(frog), 'test_images/cascade_frog.png')

    # Section 6 - Seam Carving
    # twocats = load_color_image('test_images/twocats.png')
    # save_color_image(seam_carving(twocats, 100), 'test_images/seam_carved_twocats.png')

    # Section 7 - Something of my own (darken)
    # bluegill = load_color_image('test_images/bluegill.png')
    # save_color_image(darken(bluegill, 0.5), 'test_images/dark_bluegill.png')
    
    pass
