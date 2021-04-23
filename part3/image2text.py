#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:
# (based on skeleton code by D. Crandall, Oct 2020)
#
import math
import operator

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[2] ]))

#function to read the train_txt file
def reading_the_data():
    data_list=[]
    filename=train_txt_fname
    file=open(filename, 'r')
    for fline in file:
        #parsing
        text_seen=tuple([letter for letter in fline.split()])
        data_list+=[[text_seen]]
    return data_list

#checking to see what each pixel has;
#if space-' ', increasing the #spaces by 1
#if the testing letters is the same as the train letter, increasing the hit_count by 1
#else, miss_count++
def bayes_net_comparison( test_letters):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_determined={}
    for i in TRAIN_LETTERS:
        hit_count=0
        spaces=0
        miss_count=1
        for idx in range(0, len(test_letters)):
            for img_pixel in range(0, len(test_letters[idx])):
                if train_letters[i][idx][img_pixel]==' ' and test_letters[idx][img_pixel]==' ':
                    spaces+=1
                else:
                    if train_letters[i][idx][img_pixel]== test_letters[idx][img_pixel]:
                        hit_count +=1
                    else:
                        miss_count +=1
        # still working on this probability factor
        letter_determined[' ']=0.5
        #340 because total pixels=350, so asssigning a high probability if spaces>340-based on trial and error
        if spaces >340:
            letter_determined[i]=spaces/float(14*25)
        else:
            letter_determined[i]=hit_count/float(miss_count)
    return max(letter_determined.items(), key=operator.itemgetter(1))[0]

def simple_bayes(test_letters):
    line= ''
    for each_letter in test_letters:
        line+= bayes_net_comparison(each_letter)
        # print(bayes_net_comparison(each_letter))
    return line


# The final two lines of your output should look something like this:
print("Simple: " + simple_bayes(test_letters) )
# print("   HMM: " + hmm_map(test_letters))



