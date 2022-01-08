#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 01:14:13 2021

@author: shubhamrathi
"""
import numpy as np
import csv

def process_line(lines, pos_words, neg_words, y):
    pronouns = ['i', 'me', 'mine', 'my', 'you', 'your', 'yours', 'we', 'us', 'ours']
    
    vectors = np.zeros((len(lines), 7))
    id = [""]*len(lines)
    for index, line in enumerate(lines):
        words = line.split()
        id[index] = [words[0]]
        for i in range(1, len(words)):
            w = words[i].translate({ord(c): None for c in '!@#$,.\'":?{}()'}).lower()
            vectors[index][0] += 1 if w in pos_words else 0
            vectors[index][1] += 1 if w in neg_words else 0
            vectors[index][2] = 1 if w == 'no' else vectors[index][2]
            vectors[index][3] += 1 if w in pronouns else 0
            vectors[index][4] = 1 if words[i].endswith('!') else vectors[index][4]
        vectors[index][5] = "%.2f" % np.log(len(words)-1)
        vectors[index][6] = y
    return np.concatenate((id, vectors), axis = 1)

neg_words = set()
pos_words = set()
vectors = []

with open('pos_words.txt') as file:
    lines = file.readlines()
    [pos_words.add(line.strip()) for line in lines]

with open('neg_words.txt') as file:
    lines = file.readlines()
    [neg_words.add(line.strip()) for line in lines]


with open('pos_reviews.txt') as file:
    lines = file.readlines()
    vectors = process_line(lines, pos_words, neg_words, 1)
    
with open('neg_reviews.txt') as file:
    lines = file.readlines()
    vectors = np.concatenate((vectors, process_line(lines, pos_words, neg_words, 0)))
    

with open('features.csv', 'w') as f:
    writer = csv.writer(f)
    for r in vectors:
        writer.writerow(r)


