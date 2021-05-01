#!/usr/bin/python
#
# Part1- part of speech tagging
#
# Authors: admysore- hdeshpa- machilla
# based on skeleton code by D. Crandall

import random
import math

class Solver:

    minimum=0.000000001
    part_of_speech_word_count=dict()
    parts_of_speech = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x','.']
    parts_of_speech_next = ['initial', 'adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x','.']
    parts_of_speech_frequency = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
    previous_next_pos_count = dict()
    previous_next_pos_probability = dict()
    part_of_speech_transition_count = dict()
    part_of_speech_probability = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
    next_given_previous_pos_count = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}

    emission_probabilities = dict()
    transition_probabilities = dict()

    def return_posterior_simple(self, sentence, label):
        Probabilities = 0
        for i in range(len(sentence)):
            Probabilities += math.log(self.part_of_speech_probability.get(label[i], self.minimum))
            Probabilities += math.log(self.emission_probabilities.get((sentence[i], label[i]), self.minimum))
        return Probabilities

    def return_posterior_HMM(self, sentence, label):
        Probabilities = 0
        for i in range(len(sentence)):
            if i == 0:
                Probabilities += math.log(self.emission_probabilities.get((sentence[i], label[i]), self.minimum))
                Probabilities += math.log(self.transition_probabilities.get(('initial', label[i])))
            else:
                Probabilities += math.log(self.emission_probabilities.get((sentence[i], label[i]), self.minimum))
                Probabilities += math.log(self.transition_probabilities.get((label[i - 1], label[i])))
        return Probabilities

    def return_posterior_Gibbs(self, sentence, label):
        Probabilities = 0
        for word_position in range(len(sentence)):
            if len(sentence) == 1:
                Probabilities += math.log(self.emission_probabilities.get((sentence[word_position], label[word_position]), self.minimum)) \
                     + math.log(self.transition_probabilities.get(('initial', label[word_position]), self.minimum))

            elif word_position == 0:
                Probabilities += math.log(self.emission_probabilities.get((sentence[word_position], label[word_position]), self.minimum)) \
                     + math.log(self.transition_probabilities.get(('initial', label[word_position]), self.minimum)) \
                     + math.log(self.transition_probabilities.get((label[word_position], label[word_position + 1]), self.minimum))

            elif word_position == len(sentence) - 1:
                Probabilities += math.log(self.emission_probabilities.get((sentence[word_position], label[word_position]), self.minimum)) \
                     + math.log(self.previous_next_pos_probability[label[word_position - 1], label[0]].get(label[word_position], self.minimum))

            else:
                Probabilities += math.log(self.emission_probabilities.get((sentence[word_position], label[word_position]), self.minimum)) \
                     + math.log(self.transition_probabilities.get((label[word_position - 1], label[word_position]), self.minimum)) \
                     + math.log(self.transition_probabilities.get((label[word_position], label[word_position + 1]), self.minimum))

        return Probabilities

    def posterior(self, model, sentence, label):
        if model == "Simple":
            Probabilities = self.return_posterior_simple(sentence,label)
            return Probabilities
        elif model == "HMM":
            Probabilities=self.return_posterior_HMM(sentence,label)
            return Probabilities
        elif model == "Complex":
            Probabilities = self.return_posterior_Gibbs(sentence, label)
            return Probabilities
        else:
            print("Unknown algo!")

    def initialise_previous_next_pos_count(self):
        for pos_prev in self.parts_of_speech:
            for pos_next in self.parts_of_speech:
                self.previous_next_pos_count[pos_prev, pos_next] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                                               'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
                self.previous_next_pos_probability[pos_prev, pos_next] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0,
                                              'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}

    def initialise_pos_transition_count(self):
        for pos_next in self.parts_of_speech_next:
            for pos in self.parts_of_speech:
                self.part_of_speech_transition_count[pos_next, pos] = 0

    def count_all_pos_word(self, data):
        for sentence, pos_labels in data:
            for word, labels in zip(sentence, pos_labels):
                if word in self.part_of_speech_word_count.keys():
                    self.part_of_speech_word_count[word][labels] += 1
                else:
                    self.part_of_speech_word_count[word] = {'adj': 0, 'adv': 0, 'adp': 0, 'conj': 0, 'det': 0, 'noun': 0, 'num': 0, 'pron': 0, 'prt': 0, 'verb': 0, 'x': 0, '.': 0}
                    self.part_of_speech_word_count[word][labels] += 1
                self.parts_of_speech_frequency[labels] += 1

            for label_num in range(len(pos_labels)):
                if label_num == 0:
                    self.part_of_speech_transition_count['initial', pos_labels[label_num]] += 1
                elif label_num == len(pos_labels) - 1:
                    self.part_of_speech_transition_count[pos_labels[label_num - 1], pos_labels[label_num]] += 1
                    self.previous_next_pos_count[pos_labels[label_num-1], pos_labels[0]][pos_labels[label_num]] += 1
                else:
                    self.part_of_speech_transition_count[pos_labels[label_num - 1], pos_labels[label_num]] += 1

    def calculate_prob_pos(self):
        total_pos_labels = sum([self.parts_of_speech_frequency[pos] for pos in self.parts_of_speech])
        for pos in self.parts_of_speech:
            self.part_of_speech_probability[pos] = self.parts_of_speech_frequency[pos] / total_pos_labels

    def count_next_given_previous_pos(self):
        for pos in self.parts_of_speech:
            for pos_next in self.parts_of_speech_next:
                self.next_given_previous_pos_count[pos] += self.part_of_speech_transition_count[pos_next, pos]

    def calculate_emission_probabilities(self):
        for word in self.part_of_speech_word_count.keys():
            for pos in self.parts_of_speech:
                if self.part_of_speech_word_count[word][pos] == 0:
                    self.emission_probabilities[word, pos] = self.minimum
                else:
                    self.emission_probabilities[word, pos] = self.part_of_speech_word_count[word][pos] / self.parts_of_speech_frequency[pos]

    def calculate_transition_probabilities(self):
        for (prev, next) in self.part_of_speech_transition_count.keys():
            if self.part_of_speech_transition_count[prev, next] == 0:
                self.transition_probabilities[prev, next] = self.minimum
            else:
                self.transition_probabilities[prev, next] = self.part_of_speech_transition_count[prev, next] / self.next_given_previous_pos_count[next]

        #for complex model
        self.total_counts = dict()
        for i in self.previous_next_pos_count.keys():
            self.total_counts[i] = 0

        for pos in self.previous_next_pos_count:
            for count in self.previous_next_pos_count[pos]:
                self.total_counts[pos] += self.previous_next_pos_count[pos][count]

        for pos in self.previous_next_pos_count.keys():
            for count in self.previous_next_pos_count[pos]:
                if self.total_counts[pos] == 0 or self.previous_next_pos_count[pos][count] == 0:
                    self.previous_next_pos_probability[pos][count] = self.minimum
                else:
                    self.previous_next_pos_probability[pos][count] = self.previous_next_pos_count[pos][count] / \
                                                                     self.total_counts[pos]

    def train(self, data):
        self.initialise_previous_next_pos_count()
        self.initialise_pos_transition_count()
        self.count_all_pos_word(data)
        self.calculate_prob_pos()
        self.count_next_given_previous_pos()
        self.calculate_emission_probabilities()
        self.calculate_transition_probabilities()

    def simplified(self, sentence):
        labels = []
        label = ""
        for word in sentence:
            if word in self.part_of_speech_word_count:
                max_label = 0
                for pos in self.parts_of_speech:
                    if self.part_of_speech_word_count[word][pos] > max_label:
                        label = pos
                        max_label = self.part_of_speech_word_count[word][pos]
                labels.append(label)
            else:
                labels.append("noun")
        return labels

    def build_viterbi_tables(self, sentence):
        self.viterbi_table = [[] for pos_count in range(len(self.parts_of_speech_frequency.keys()))]
        self.viterbi_route_for_max = [[] for pos_count in range(len(self.parts_of_speech_frequency.keys()))]
        iterate = 0
        for word in sentence:
            if iterate == 0:
                for index in range(0, len(self.parts_of_speech)):
                    word_occ = math.log(
                        self.emission_probabilities.get((word, self.parts_of_speech[index]), self.minimum)) + \
                                math.log(self.transition_probabilities.get(('initial', self.parts_of_speech[index]),
                                                                           self.minimum))
                    self.viterbi_table[index].append(word_occ)
            else:
                for index in range(0, len(self.parts_of_speech)):
                    em_prob = math.log(
                        self.emission_probabilities.get((word, self.parts_of_speech[index]), self.minimum))
                    max_prev_state = []

                    for j in range(0, len(self.parts_of_speech)):
                        trans_prob = self.transition_probabilities.get(
                            (self.parts_of_speech[j], self.parts_of_speech[index]), self.minimum)
                        temp = self.viterbi_table[j][iterate - 1] + math.log(trans_prob)
                        max_prev_state.append(temp)

                    max_val = max(max_prev_state)
                    max_index = max_prev_state.index(max_val)
                    self.viterbi_route_for_max[index].append(max_index)
                    self.viterbi_table[index].append(em_prob + max_val)
            iterate += 1

    def hmm_viterbi(self, sentence):
        self.build_viterbi_tables(sentence)
        rows_of_table = []

        for r in self.viterbi_table:
            rows_of_table.append(r[len(r) - 1])
        indexes = rows_of_table.index(max(rows_of_table))
        seq = [indexes]
        count = len(self.viterbi_route_for_max[0]) - 1
        while count >= 0:
            temp = self.viterbi_route_for_max[indexes][count]
            seq.append(temp)
            indexes = temp
            count -= 1
        seq.reverse()
        labels = [self.parts_of_speech[s] for s in seq]
        return labels

    def build_sequence_counts_samping_and_healing_periods(self, sentence, total_samples, healing_period):
        self.count_dict = {}
        self.sequence_counts = ['noun'] * len(sentence)

        for index in range(len(sentence)):
            self.count_dict[index] = {}
            for pos in self.parts_of_speech_frequency.keys():
                self.count_dict[index][pos] = 0

        for s in range(len(self.sequence_counts)):
            self.count_dict[s][self.sequence_counts[s]] += 1

        for samples in range(total_samples):
            for s in range(len(self.sequence_counts)):
                if len(sentence) == 1:
                    Probabilties = [math.log(self.emission_probabilities.get((sentence[s], k), self.minimum))
                                    + math.log(self.transition_probabilities.get(('initial', k), self.minimum)) for k in
                                    self.parts_of_speech]
                elif s == 0 :
                    Probabilties = [math.log(self.emission_probabilities.get((sentence[s], k), self.minimum))
                                    + math.log(self.transition_probabilities.get(('initial', k), self.minimum))
                                    + math.log(
                        self.transition_probabilities.get((k, self.sequence_counts[s + 1]), self.minimum)) for k in
                                    self.parts_of_speech]
                elif s == len(self.sequence_counts) - 1:
                    Probabilties = [math.log(self.emission_probabilities.get((sentence[s], k), self.minimum))
                                    + math.log(
                        self.previous_next_pos_probability[self.sequence_counts[s - 1], self.sequence_counts[0]].get(k,
                                                                                                           self.minimum))
                                    for k in self.parts_of_speech]
                else:
                    Probabilties = [math.log(self.emission_probabilities.get((sentence[s], k), self.minimum))
                                    + math.log(
                        self.transition_probabilities.get((self.sequence_counts[s - 1], k), self.minimum))
                                    + math.log(
                        self.transition_probabilities.get((k, self.sequence_counts[s + 1]), self.minimum)) for k in
                                    self.parts_of_speech]
                Probabilties = [math.exp(Probabilties[i]) for i in range(len(Probabilties))]
                total = sum(Probabilties)
                Probabilties = [Probabilties[i] / total for i in range(len(Probabilties))]
                rand = random.random()
                c = 0
                for p_count in range(len(Probabilties)):
                    c += Probabilties[p_count]
                    if rand < c:
                        self.sequence_counts[s] = self.parts_of_speech[p_count]
                        break

                if samples > healing_period:
                    for k in range(len(self.sequence_counts)):
                        self.count_dict[k][self.sequence_counts[k]] += 1

    def complex_mcmc(self, sentence):
        labels = []
        self.build_sequence_counts_samping_and_healing_periods(sentence, 1000, 550)
        for s in range(len(self.sequence_counts)):
            pos = ""
            max_count = 0
            for pos in self.parts_of_speech_frequency.keys():
                if self.count_dict[s][pos] >= max_count:
                    max_count = self.count_dict[s][pos]
                    partofspeech = pos
            labels.append(partofspeech)
        return labels

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")
