#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: admysore- hdeshpa- machilla
# (based on skeleton code by D. Crandall, Oct 2020)
#
import math
import operator

from PIL import Image, ImageDraw, ImageFont
import sys

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                    range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


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

# function to read the train_txt_fname file
def reading_the_data():
    data_list = []
    filename = train_txt_fname
    file = open(filename, 'r')
    for fline in file:
        # parsing
        text_seen = tuple([letter for letter in fline.split()])
        data_list += [[text_seen]]
    return data_list

# checking to see what each pixel has;
# if space-' ', increasing the count of spaces by 1
# if the test letters is the same as the train letter, increasing the character_match by 1
# else, character_mismatch++
def simple_probability_calculation(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_determined = {}
    for i in TRAIN_LETTERS:
        character_match = 0
        spaces_encountered = 0
        character_mismatch = 1
        for idx in range(0, len(test_letters)):
            for img_pixel in range(0, len(test_letters[idx])):
                if train_letters[i][idx][img_pixel] == ' ' and test_letters[idx][img_pixel] == ' ':
                    spaces_encountered += 1
                else:
                    if train_letters[i][idx][img_pixel] == test_letters[idx][img_pixel]:
                        character_match += 1
                    else:
                        character_mismatch += 1
        letter_determined[' '] = 0.4
        # 340 because total pixels=350
        #we tried it for other values, some of it gives an excellent output for a few images while poor results for the other images
        if spaces_encountered > 340:
            #14*25=350
            letter_determined[i] = spaces_encountered / float(350)
        else:
            letter_determined[i] = character_match / float(character_mismatch)
    # return character_match, spaces_encountered, character_mismatch, letter_determined
    return letter_determined

# read the training data
# transition_state_dictionary-transition probability for all possible letters in training
def transition_probability():
    data_list = reading_the_data()
    transition_state_dictionary = {}
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for word in data_list:
        str = (" ").join(word[0])
        for i in range(0, len(str) - 1):
            #if already in the transition state dictionary, incrementing by 1
            if (str[i] in TRAIN_LETTERS) and (str[i + 1] in TRAIN_LETTERS) and transition_state_dictionary.__contains__(str[i] + "$" + str[i + 1]):
                transition_state_dictionary[str[i] + "$" + str[i + 1]] = transition_state_dictionary[str[i] + "$" + str[i + 1]] + 1
            else:
                #else, setting the value to 1
                if (str[i] in TRAIN_LETTERS) and (str[i + 1] in TRAIN_LETTERS):
                    transition_state_dictionary[str[i] + "$" + str[i + 1]] = 1
    transitions_sum_dictionary = {}
    for i in range(0, len(TRAIN_LETTERS)):
        probabilitysum = 0
        for key in transition_state_dictionary.keys():
            if (TRAIN_LETTERS[i] == key.split('$')[0]):
                probabilitysum += transition_state_dictionary[key]
        if probabilitysum != 0:
            transitions_sum_dictionary[TRAIN_LETTERS[i]] = probabilitysum
    for key in transition_state_dictionary.keys():
        transition_state_dictionary[key] = (transition_state_dictionary[key]) / (float(transitions_sum_dictionary[key.split("$")[0]]))
    # print(trasition_state_dictionary)
    totalprobability = sum(transition_state_dictionary.values())
    for key in transition_state_dictionary.keys():
        transition_state_dictionary[key] = transition_state_dictionary[key] / float(totalprobability)
    return data_list,transition_state_dictionary
    # print(trasition_state_dictionary)

# frequency of first character- for each word in each image, we calculate the frequency of the 1st character/letter of each word
# we also calculate the frequency of each character
#how likely each observed state is, given being in a certain hidden state.
def emission_probability():
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    each_img, transition_state_dictionary=transition_probability()
    frequencies_of_first_character = {}
    # each img
    for each_data in each_img:
        # each word
        for word in each_data[0]:
            # print(word)
            #each letter
            if word[0] in TRAIN_LETTERS:
                #if already in dictionary, increment by 1
                if frequencies_of_first_character.__contains__(word[0]):
                    frequencies_of_first_character[word[0]] += 1
                else:
                    #else, setting the value to 1
                    frequencies_of_first_character[word[0]] = 1
    # frequency of all the characters
    totalprobability = sum(frequencies_of_first_character.values())
    for key in frequencies_of_first_character.keys():
        frequencies_of_first_character[key] = frequencies_of_first_character[key] / float(totalprobability)
    # print(frequencies_of_first_character)
    numberofcharacters = 0
    #frequency of each character
    each_character_frequency = {}
    for word in each_img:
        str = (" ").join(word[0])
        for char in str:
            if char in TRAIN_LETTERS:
                numberofcharacters = numberofcharacters + 1
                if each_character_frequency.__contains__(char):
                    each_character_frequency[char] += 1
                else:
                    each_character_frequency[char] = 1
    # print(each_character_frequency)
    # print(numberofcharacters)
    for key in each_character_frequency.keys():
        each_character_frequency[key] = (each_character_frequency[key] ) / (float(numberofcharacters) + math.pow(10, 10))
    # print(each_character_frequency)
    totalprobability = sum(each_character_frequency.values())
    for key in each_character_frequency.keys():
        each_character_frequency[key] = each_character_frequency[key] / float(totalprobability)
    # print(each_character_frequency)
    return [ frequencies_of_first_character,each_character_frequency, transition_state_dictionary]

#for each test_letter, we calculate the hit and miss ratio and take the letter which has the highest probability
def simple_bayes_net(test_letters):
    output_word = ''
    for each_letter in test_letters:
        letter = simple_probability_calculation(each_letter)
        max_probability_letter = max(letter.items(), key=operator.itemgetter(1))[0]
        output_word += max_probability_letter
        #print(output_word)
    return output_word

#for each letter determined, we calculate the probability
def highest_probable_letter(test_letters):
    # for each letter in test_letters:
    letter_determined = simple_probability_calculation(test_letters)
    # print(letter_determined)
    probability_total = 0
    for key in letter_determined.keys():
        if key != " ":
            probability_total = probability_total + letter_determined[key]
        else:
            probability_total = probability_total + 1
    for key in letter_determined.keys():
        if key != " ":
            if letter_determined[key] != 0:
                letter_determined[key] = letter_determined[key] / float(probability_total)
            else:
                letter_determined[key] = 0.002
    return letter_determined

#we take the 4 most probable letters
def most_probable_letters(test_letters):
    letter_determined=highest_probable_letter(test_letters)
    returnLetter = dict(sorted(letter_determined.items(), key=operator.itemgetter(1), reverse=True)[:4])
    return returnLetter

#the most probable letter determined for the given the test letter
def initial_probability(test_letters):
    initial_probability = most_probable_letters(test_letters[0])
    return initial_probability

#We get the initial probability using the most_probable_letters function.
#Using the transition and emission probabilities, we calculate the probability of each letter and store it.
#taking log probabilities to handle underflow
def hmm(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    listofchars = ['']*len(test_letters)
    vMat_map = []
    for i in range(0, len(TRAIN_LETTERS)):
        temporary = []
        for j in range(0, len(test_letters)):
            temporary.append([0,''])
        vMat_map.append(temporary)
    init_probability = initial_probability(test_letters)
    # print(init_probability)
    for row in range(0,len(TRAIN_LETTERS)):
        if TRAIN_LETTERS[row] in init_probability and init_probability[TRAIN_LETTERS[row]]!=0 and TRAIN_LETTERS[row] in frequencies_of_first_character:
            vMat_map[row][0] = [- math.log10(init_probability[TRAIN_LETTERS[row]]),'q1']
    for col in range(1,len(test_letters)):
        letter_determined = most_probable_letters(test_letters[col])
        # print(letter_determined)
        if ' ' in letter_determined:
            listofchars[col] = " "
        for key in letter_determined.keys():
            temporary = {}
            for row in range(0,len(TRAIN_LETTERS)):
                if key in letter_determined and (TRAIN_LETTERS[row]+"$"+key) in trasition_state_dictionary:
                    temporary[TRAIN_LETTERS[row]] = 0.0002 * vMat_map[row][col-1][0]- math.log10(trasition_state_dictionary[TRAIN_LETTERS[row]+"$"+key])- 10 * math.log10(letter_determined[key])
            max = 0
            Maxkey = ''
            for i in temporary.keys():
                if max < temporary[i]:
                    max = temporary[i]
                    Maxkey = i
            if Maxkey != '':
                vMat_map[TRAIN_LETTERS.index(key)][col] = [temporary[Maxkey],Maxkey]
    return vMat_map,listofchars

#appending the results to a string
def map_hmm_calculation(test_letters):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    vMat_map,listofchars =hmm(test_letters)
    maximum = math.pow(10, 10)
    for row in range(0, len(TRAIN_LETTERS)):
        if maximum > vMat_map[row][0][0] and vMat_map[row][0][0] != 0:
            maximum = vMat_map[row][0][0]
            listofchars[0] = TRAIN_LETTERS[row]
    for col in range(1, len(test_letters)):
        minimum = math.pow(10, 10)
        for row in range(0, len(TRAIN_LETTERS)):
            if vMat_map[row][col][0] < minimum and vMat_map[row][col][0] != 0 and row != len(TRAIN_LETTERS)-1 and listofchars[col]!=' ':
                minimum = vMat_map[row][col][0]
                listofchars[col] = TRAIN_LETTERS[row]
    return "".join(listofchars)

# The final two lines of your output should look something like this:
[frequencies_of_first_character, each_character_frequency, trasition_state_dictionary] = emission_probability()
print("Simple: " + simple_bayes_net(test_letters))
print("   HMM: " + map_hmm_calculation(test_letters))



