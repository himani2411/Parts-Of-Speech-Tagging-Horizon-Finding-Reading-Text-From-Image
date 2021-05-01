Assignment 3

Part 1- Part of speech tagging

Approach:

The aim of this program is to be able to train and tag each word of a sentence to one of the 12 parts of speech which include Adjective, Adverb, Adposition, Conjunctive, Determiner, Noun, Number, Pronoun, Particle, Verb, Foreign word, and punctuation mark.

First we train, in which we initialise and calculate all the prior counts and probabilities and also calculate the emission and transitional probabilities throughout the entire train set.

We have been asked to implement three types of algorithms:

-\&gt;simple

-\&gt;HMM

-\&gt;complex

Simple:

In the simple approach, we were asked to implement Bayes net. In this approach, the probability of part of speech given word was calculated for all part of speeches and the maximum of those was considered as final label for that word. This probability can be either be the emission probability or simply the maximum of word count as implemented in the code. Here It is assumed that each part of speech is independent from the other.

HMM:

For this, we were asked to implement a Bayes net, to implement Viterbi algorithm to find MAP (Maximum a Posteriori). In this Viterbi algorithm, we implemented a function to build the Viterbi table and a table to record the maximum route. The words of the sentence are observed variables and parts of speeches are the hidden variables. We calculate word given part of speech using the transition and emission probabilities for all the parts of speeches and take the maximum value of all, this value is assigned to the Viterbi table. The route this algorithm takes is stored at every step, and new part of speech is added to the Viterbi route. After building the dictionaries for Viterbi and max route, in the hmm\_viterbi function we procure the most likely sequence of part of speeches to occur, and return a list of labels for those words in the sentence.

Complex:

For complex approach, we were asked to implement Gibbs sampling, one of the difference in this approach from the above approach is that now, this is a tougher Bayeys net in which with every part of speech instead of just one, 2 parts of speech are considered, the previous and the next. With the help of emission and transition, probabilities are calculated based on word position in the sentence and kept for samples, once the healing period is reached, the maximum and the part of speech of these probabilities is captured in count\_dict and sequence\_counts dictionaries. In the main complex\_mcmc function, we iterate through the sequence\_counts, take the values of maximum occurring part of speech for each word and store it in labels, and return this list of labels for the sentence sent to this function.

For the posterior probabilities:

For the simple approach, we are simply calculating the posterior as probability of word given part of speech multiplied by probability of obtaining that part of speech, for all the words in the sentence. For the HMM Viterbi approach, we are calculating the probability of word given part of speech multiplied by probability of current part of speech given previous part of speech for all the words in the sentence. And lastly, for the complex approach we are multiplying probability of word given part of speech, current part of speech given previous part of speech and current part of speech given next part of speech for all the words in the sentence.

Small note, instead of dealing directly with probabilities, throughout the entire program, we have converted it to logs, and done arithmetic operations accordingly. we have set a minimum variable to 0.000000001(a very low value), to avoid multiplication or division with 0, or cover for any missing words, or zero probabilities etc.

Functions:

-\&gt;def posterior(self, model, sentence, label):

Function to calculate posterior probabilities for all three approaches.

helper functions:

- def return\_posterior\_simple(self, sentence, label):

Returns probabilities, calculated according to simple approach stated above

- def return\_posterior\_HMM(self, sentence, label):

Returns probabilities, calculated according to HMM viterbi approach stated above

- def return\_posterior\_Gibbs(self, sentence, label):`

Returns probabilities, calculated according to Gibbs sampling approach stated above

-\&gt;def train(self, data):

Function to train our data, collect all the pre requisite counts and probabilities, including transition and emission probabilities throughout the train set, before implementing three approaches.

helper functions:

- def initialise\_previous\_next\_pos\_count(self):

to initialise dictionary that keeps a track of all 2 pair combinations of parts of speech as previous and next, later to count the frequency of each possible pair.

- def initialise\_pos\_transition\_count(self):

to initialise dictionary that keeps a track of count of transitions of parts of speech of words.

- def count\_all\_pos\_word(self, data):

goes through every word and keeps count of each of the part of speech encountered.

- def calculate\_prob\_pos(self):

to calculate the probability of each of the part of speech.

- def count\_next\_given\_previous\_pos(self):

Counting the appearance of each of the part of speech given a previous part of speech.

- def calculate\_emission\_probabilities(self):

this functions is to calculate and store all emission probabiltities of each word, given part of speech is nothing but count of part of speech for that word by the total frequency of that that part of speech.

- def calculate\_transition\_probabilities(self):

this function is to calculate and store all transitional probabilities which is calculated for all combinations of parts of speech.

-\&gt;def simplified(self, sentence):

Function to implement Simple approach of bayes nets

-\&gt;def hmm\_viterbi(self, sentence):

Function to implement hidden markov model, Viterbi algorithm .

helper function:

- def build\_viterbi\_tables(self, sentence):

This function is implemented to construct the Viterbi tables, taking into account previous and current part of speech, its emission, transition probabilities and to store the route of sequence of part of speeches that has the maximum probability values.

-\&gt;def complex\_mcmc(self, sentence):

Function to implement Gibbs sampling.

helper function:

- def build\_sequence\_counts\_samping\_and\_healing\_periods(self, sentence, total\_samples, healing\_period):

for total\_samples, with help of transition and emission, probabilities and total counts are calculated and stored, at the healing period, these counts are updated. The counts\_dict and sequence\_counts are updated for group of samples, and dictionaries used in the main complex\_mcmc function to return final lables for a sentence


So far scored 2000 sentences with 29442 words.

                   Words correct:     Sentences correct:

   0. Ground truth:      100.00%              100.00%

         1. Simple:       93.95%               47.50%

            2. HMM:       95.09%               54.40%

        3. Complex:       95.08%               54.05%
    

Part 2
Approach 1 - Bayes Net

We loop through all the rows of the edge strength matrix given by the edge_strength function to get all the values of a single column. We store all these intensity values in a list.
And take the maximum value of this list. And find the index of this maximum Value. then store this in the result list( ridge_line ), which is storing all the y co-orinates of the image.
In a nut shell, we are getting the maximum intensity value of each column and storing its row index (y co-ordinate ) in a result.

Then we highlight this ridgeline using the draw_edge function with red color.

Approach 2 - Viterbi
We get the initial probabilities using the get_initial_pixel_probability function.
In this function, I am creating a table which will store all the probabilities of all the pixel at the same location indexes as that of edge strength matrix.
So this table, is first initialized with zeros. We get the sum of all the intensity values of a single column. Then for each pixel 
in that column we get their probability by dividing it by the sum that we calculated for that column.
The initial probability will be calculated for all the pixel in the first column of the image.

As mentioned in the assignment we have assumed the following transition probabilities so that there is a smoothness when we transition from on pixel to next.
We have kept the transition probability for the pixel in the same row as high(0.8) as there will be a greater change of the ridge line being in the same row or just the a few rows above or below.
We keep on decreasing the transition probabilities as we move up or below from the pixel for which we are calculating the probabilities.
For all the other rows we are not calculating their probabilities as the chances of them being the ridgeline will reduce.
For example, if the ridge line starts at row 50 and column 50, then we will look at the above 5 and below 5 rows. So we will check from row 45 to 55.
For example, We wont check row 0, 10, 36, 100, 150, etc as the chances
 of an edge to move from row 50 col 50 (50,50) to row 100 and column 51 (51, 100) will be low if not negligible.
 
We tried a different combinations of rows and probabilities but found that these give overall good results for all the images.
[0.8, 0.5, 0.2, 0.05, 0.005, 0.0005]

The get_transition_probability() function will fill up the probability table for the 5 rows above and below the row we are looking at and then continue to the next column.
So the outter two for loops will loop for all the rows and columns we have in the image. The inner for loop I have kept at a range so that it will look at the top and bottom 5 rows only.
We keep track of the maximum probability that we have encountered during the loop and store their location in a table (previous_max_pixel_table) which will help us when we want to backtrack.

We find the maximum probability for all the rows in the final column of the image and store it in maximum_intensity_pixel_prob.
So we backtrack from the final column to the first column of the image and get all the pixel locations using the previous_max_pixel_table table where we have stored the locations.
We store these locations for all the columns in the second_ridgeline list and draw a blue edge over the original image


Approach 3 - Human Input and Viterbi
We use the row and column given and reset the row and column values of the probabilities and then calculate these probabilities.
The first get_transition_probability() function call is so that we traverse from the human input row and column towards the first column of the image. 
The second get_transition_probability() function call is so that we traverse from the given input row and column to the last column of the image.

After these probabilities are newly calculated for these human input column and rows we  backtrack and then find the new ridgeline as done in the previous approach.
We store these locations for all the columns in the third_ridgeline list and draw a green edge over the original image
   
 




References:

https://en.wikipedia.org/wiki/Image_gradient

https://www.cse.unr.edu/~bebis/ISVC13_Horizon.pdf

https://www.cse.unr.edu/~bebis/ICMLA15.pdf

https://apps.dtic.mil/dtic/tr/fulltext/u2/1029726.pdf

https://core.ac.uk/download/pdf/10882228.pdf

https://ntrs.nasa.gov/api/citations/20160011500/downloads/20160011500.pdf

http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm


PART 3- Reading text

Approach-

1. Simple bayes net

Here, we&#39;re checking what is stored in each pixel. We know that black dots are represented by \*&#39;s and white dots are spaces. We compare the train and test characters to see if they match or not. We keep count of all the matches and mismatches and the number of spaces encountered.

For each character in the image, we take the character from a dictionary of characters that which has the highest probability.

We keep appending each such character to a string and return the final output.

1. HMM-MAP

HMM is a special case of bayes net wherein we find out the initial, transition and the emission probabilities.

Here, given a particular observation, we&#39;re trying to find the most likely sequence of letters across all the images.

We try to maximize P(Q/O)( which is nothing but a product of initial probability \* product of transition probabilities and product of emission probabilities. Using these 3 probabilities, we calculate the probability of each letter and store it. We do it for each character, where we pick the one with the highest probability. Each value in the matrix is calculated from the previous column&#39;s value. We do this for all the characters present in an image, normalize the values by taking log probabilities and append it to the output string

Functions-

reading\_the\_data()-

Here, we&#39;re reading the train\_text.txt file (train\_txt\_fname). This function opens the text file and parse the words and store it in a list.

We return a list of words.

hit\_miss\_calculation(test\_letters)-

It takes 1 parameter- test letters.

Here, we&#39;re checking what is stored in each pixel.

We compare the train and the test letters and if a space &#39; &#39; is encountered, we increase the space count by 1.

If the train and the test letters are the same, then we increase the count of the hit count.

If not, we increase the count of the miss count.

For spaces, we assign a probability of 0.4 based on trial and error.

As the total number of pixels is 350(14\*25), if the space count is greater than 340, we assign it a high probability, else, the letter determined would have a ratio of hit count/ miss count.

Here, we tried it for other values than 340. For some values, the output of the first few images were nearly perfect but it gave poor results for the other images.

We return a dictionary of letter determined.

transition\_probability:

Here, we&#39;re calculating the transition probability for all the letters in training data.

First, we read the training data and maintain a dictionary.

For every word in every image, we see if a letter and it&#39;s succeeding letter is already in the dictionary. If so, we increment the value by 1 else we set the value to 1.

For every entry in the dictionary, we normalize the value by taking the sum of all the values and dividing each value by the total.

We return the word list and the transition state dictionary.

emission\_probability:

Here, we&#39;re calculating the frequency of the first character in every word in every image and also the frequency of each character. We see how likely each observed state is, given being in a certain hidden state.

As done in transition probability, we maintain dictionaries and see if the entry is already in the dictionary. If so, we increment the value by 1 else we set the value to 1.

We again normalize these values and return the transition state dictionary, frequency of the first letter and the frequency of each character.

simple\_bayes(test\_letters):

It takes 1 parameter- test\_letters.

For each test\_letter, we calculate the hit and miss ratio and take the letter which has the highest probability

Each such determined letter is appended to a string and at the end, we return the output word.

highest\_probable\_letter(testLetter):

Here, we call the hit\_miss\_calculation function, which gives the determined letter.

For each such determined letter, we calculate the probability.

For spaces, we used a probability value of 0. 002 by trial and error.

This returns a dictionary of possible letters, which has the newly calculated probabilities.

most\_probable\_letters(testLetter):

It takes 1 parameter- testLetter.

Here, we call highest\_probable\_letter function and get the dictionary of possible letters.

We return the top 4 most probable letters.

initial\_probability(test\_letters):

It takes 1 parameter- testLetter.

The most probable letter is determined for the given the test letter

And returns the initial probability.

hmm(test\_letters):

It takes 1 parameter- testLetter.

Here, we maintain a list equal to the length of the test letters.

We get the initial probability using the most\_probable\_letters function.

Using the initial, transition and emission probabilities, we calculate the probability of each letter and store it.

For each letter prediction, we pick the letter which has the highest probability.

We return the list of all such characters.

map\_hmm\_calculation(test\_letters):

It takes 1 parameter- testLetter.

Here, we&#39;re appending the results to a string

For training, we've made a file called 'train-text.txt' which comprises of all the sentences given in the test images.

REFERENCES-

Artificial Intelligence- A modern approach- Peter Norvig

Professor Crandall&#39;s slides and lecture videos.

[https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e](https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e)

[https://papers.nips.cc/paper/2011/file/c8c41c4a18675a74e01c8a20e8a0f662-Paper.pdf](https://papers.nips.cc/paper/2011/file/c8c41c4a18675a74e01c8a20e8a0f662-Paper.pdf)
