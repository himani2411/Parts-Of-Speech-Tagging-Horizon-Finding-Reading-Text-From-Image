#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#

from PIL import Image
from numpy import *
# from numpy.core import amax
from numpy.ma import amax
from scipy.ndimage import filters
import sys
import imageio
from numpy import array, zeros, sqrt, where


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):

    grayscale = array(input_image.convert('L'))
    # imageio.imwrite('grayscale.jpg', grayscale)
    # print("Original imange", len(grayscale))
    filtered_y = zeros(grayscale.shape)

    filters.sobel(grayscale,0,filtered_y)
    # print("filters", sqrt(filtered_y**2))
    # imageio.imwrite('test.jpg', filtered_y)
    return sqrt(filtered_y**2)

#
# def edge_strength_test(input_image):
#
#     grayscale = array(input_image.convert('L'))
#     # imageio.imwrite('grayscale.jpg', grayscale)
#     # print("grayscale", grayscale)
#     filtered_y = zeros(grayscale.shape)
#     filtered_x = zeros(grayscale.shape)
#
#     filters.sobel(grayscale,0,filtered_y)
#     filters.sobel(grayscale,0,filtered_x)
#     # print("filters", sqrt(filtered_y**2))
#     imageio.imwrite('test1.jpg', uint8(sqrt(filtered_y**2 + filtered_x**2)))
#     # return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image


# main program
#
gt_row = -1
gt_col = -1
if len(sys.argv) == 2:
    input_filename = sys.argv[1]
elif len(sys.argv) == 4:
    (input_filename, gt_row, gt_col) = sys.argv[1:]
else:
    raise Exception("Program requires either 1 or 3 parameters")

# load in image
input_image = Image.open(input_filename)



# print(type(input_image))


# compute edge strength mask
edge_strength = edge_strength(input_image)
# edge_strength_test(input_image)
# print(len(edge_strength))    #Length is 141
# print(list(edge_strength))    #Gives a list of numpy arrays so rows are numpy arrays
print("type(edge_strength)",type(edge_strength))
# print(edge_strength[0][1])    #Gives a list of numpy arrays so rows are numpy arrays
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
# ridge =  [ ((edge_strength.shape[1]/edge_strength.shape[0])*   (edge_strength.shape[0]/edge_strength.shape[1])/edge_strength.shape[1])]
# ridgeLine | columns = ((columns | ridgeLine) * (RidgeLine))/ Columns
# print("ridge",ridge)
# print("edge_strength.shape[0] ", edge_strength.shape[0])
# [(141, 251)]   (row, Column)


#
#
# First Approach-Bayes Net
# print("edge_strength[0]", edge_strength[0])
# Fetching
ridge_line =[]
(row_len , col_len) = edge_strength.shape
for i in range(0, col_len):

    single_row = edge_strength[0:row_len, i].flatten()
    single_row = list(single_row)
    highest_intensity_col = single_row.index(max(single_row))
    ridge_line.append(highest_intensity_col)

# print(ridge_line)
# print(single_row, len(single_row))
# print(max(single_row))
# print(highest_intensity_col)
# output answer
imageio.imwrite("output_first.jpg", draw_edge(input_image, ridge_line, (0, 0, 255), 5))
# output answer
# imageio.imwrite("output.jpg", draw_edge(input_image, ridge, (255, 0, 0), 5))
