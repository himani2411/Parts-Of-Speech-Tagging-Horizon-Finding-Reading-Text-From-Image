#!/usr/local/bin/python3
#
# Authors: admysore-hdeshpa-machilla
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#


from PIL import Image
from numpy import *
# from numpy.core import amax
from numpy.ma import amax, argmax
from scipy.ndimage import filters
import sys
import imageio
from numpy import array, zeros, sqrt, where


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):

    grayscale = array(input_image.convert('L'))

    filtered_y = zeros(grayscale.shape)

    filters.sobel(grayscale,0,filtered_y)

    return sqrt(filtered_y**2)

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


# Getting an array which will store the probabilities for every pixel.
# In this function we will only get the probability of first column in the array
def get_initial_pixel_probability(edge_strength, total_row_len, total_col_len):
    #Array which will stores the probabilities
    initial_pixel_prob = zeros(edge_strength.shape)
    column_total = zeros(total_col_len)
    # sums up the edge strength values per column
    for col in range(total_col_len):
        column_total[col] = sum(edge_strength[0:total_row_len, col])

    # calculating the initial probabilities. i.e Only first column of the image
    for row in range(total_row_len):
        initial_pixel_prob[row][0] = edge_strength[row][col] / column_total[0]

    return  initial_pixel_prob

#Calculating probability of each pixel where we have set a value for checking only a few rows up and
# below the pixel to check its worth of being a Earth Sky horizon
def get_transition_probability(probability_table, previous_max_pixel_table , total_row_len, total_col_len, row_start, row_end, row_step,
                          col_start,  col_end, col_step):

    for col in range(col_start,  col_end, col_step):
        for row in range(row_start, row_end, row_step):
            maximum_intensity_pixel_prob = 0
            for j in range(-5, 6, 1):
                if ((row + j < total_row_len) & (row + j >= 0)):
                    if (maximum_intensity_pixel_prob < probability_table[row + j][col - col_step] * (transition_probabilities[abs(j)])):
                        maximum_intensity_pixel_prob = (probability_table[row + j][col - col_step]) * (transition_probabilities[abs(j)])
                        # array which will be useful in backtracking
                        previous_max_pixel_table[row][col] = row + j
                    probability_table[row][col] = (edge_strength[row][col] / 100) * (maximum_intensity_pixel_prob)

    return (probability_table, previous_max_pixel_table )


# Function is useful in backtracking to the previous columns maximum value which is stored in array during the transition probability
def backtracking(ridge_line, maximum_intensity_pixel_prob, previous_max_pixel_table, total_col_len):
    for col in range(total_col_len - 1, -1, -1):
        ridge_line[col] = int(maximum_intensity_pixel_prob)
        maximum_intensity_pixel_prob = previous_max_pixel_table[int(maximum_intensity_pixel_prob)][col]

    return (ridge_line, previous_max_pixel_table)



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

gt_row, gt_col = int(gt_row), int(gt_col)


# compute edge strength mask
edge_strength = edge_strength(input_image)



imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

#
#
# First Solution : Bayes Net

ridge_line =[]
(total_row_len, total_col_len) = edge_strength.shape
for i in range(0, total_col_len):
    #Getting all the values of a column in a list
    single_row = edge_strength[0:total_row_len, i].flatten()
    single_row = list(single_row)
    # Getting the maximum value ( intensity value) of a single column and getting the index (row index)
    # at which it will have highest intensity in the image
    highest_intensity_col = single_row.index(max(single_row))
    # Appending this in a list to keep track of the row co-ordinates( y)
    ridge_line.append(highest_intensity_col)


imageio.imwrite("output_first.jpg", draw_edge(input_image, ridge_line, (255, 0, 0), 5))

#Second Solution : Viterbi Approach


# Assumed transition probability values
# the pixel in same row has high chance of bring in the ridgeLine so I have kept the transition probability as 0.8
# if pixel is in a row above or below we keep on decreasing the transition probabilities
# for all the other rows we will not calculate their probabilities as the chance of them being in the ridge line are low
transition_probabilities = [0.8, 0.5, 0.2, 0.05, 0.005, 0.0005]

initial_pixel_prob = get_initial_pixel_probability(edge_strength, total_row_len, total_col_len)


previous_max_pixel_table = zeros((total_row_len, total_col_len))
(probability_table, previous_max_pixel_table ) = get_transition_probability(initial_pixel_prob, previous_max_pixel_table ,total_row_len, total_col_len, 0, total_row_len, 1, 1, total_col_len, 1)


# MAximum intensity pixel is stored
maximum_intensity_pixel_prob = argmax(probability_table[:, total_col_len - 1])
second_ridge = zeros(total_col_len)
(second_ridge, previous_max_pixel_table ) = backtracking(second_ridge, maximum_intensity_pixel_prob, previous_max_pixel_table, total_col_len)

input_image = Image.open(input_filename)

imageio.imwrite("output_second.jpg", draw_edge(input_image, second_ridge, (0, 0, 255), 5))

# Third Solution: Taking human input for rows and Columns With viterbi
third_ridge = [total_row_len / 6] * total_col_len


# using human input values to reset the probabilities
probability_table[0:total_row_len,gt_col] = 0
probability_table[gt_row][gt_col] = 1

# checking and calculating the probabilities in backward motion
(state_probab, previous_max_pixel_table ) = \
    get_transition_probability(probability_table, previous_max_pixel_table, total_row_len,total_col_len,(total_row_len - 1), -1, -1,(gt_col - 1), 0, -1)

# checking and calculating the probabilities in forward motion
(state_probab, previous_max_pixel_table ) = get_transition_probability(state_probab, previous_max_pixel_table ,total_row_len,total_col_len, 0, total_row_len, 1,gt_col + 1, total_col_len, 1)

# backtracking to find solution
maximum_intensity_pixel = argmax(state_probab[:, total_col_len - 1])
(third_ridge, previous_max_pixel_table ) = backtracking(third_ridge, maximum_intensity_pixel, previous_max_pixel_table , total_col_len)

input_image = Image.open(input_filename)
imageio.imwrite("output_third.jpg", draw_edge(input_image, third_ridge, (0, 255, 0), 5))
