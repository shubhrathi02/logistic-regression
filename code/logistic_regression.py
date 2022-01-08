#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 03:34:30 2021

@author: shubhamrathi
"""

import csv
from math import e


def get_train_test_data(d, random_state = 42, ratio=0.8):
    data = d * 1
    n = len(d)
    i = random_state
    train_values = 0
    train = []
    test = []
    while train_values < ratio * n:
        i = i % len(data)
        train.append(data[i])
        data.remove(data[i])
        i += random_state
        train_values += 1
    
    test = data
    return train, test
    

class LogisticRegression:
    
    def __init__(self, eeta=0.1):
        self.w = None
        self.b = 0
        self.eeta = eeta
        
    
    def fit(self, train):
        self.nf = len(train[0])-2
        self.w = [0] * (self.nf+1)
        for x in train:
            loss = self.get_loss(x[1:])
            for i,w in enumerate(self.w):
                self.w[i] -= self.eeta * loss[i]
            #print(self.w)
            
            
    def calculate_sigma(self, x):
        wxb = 0
        n = len(x)
        for i in range(0, n):
            wxb += self.w[i]*float(x[i])
        #wxb += self.b
        return 1/(1 + pow(e, -wxb))
        
    
    def get_loss(self, train):
        n = len(train)
        x = train[:n-1]
        x.append('1.0')
        sigma = self.calculate_sigma(x)
        
        gradient = sigma - float(train[n-1])
        return [gradient*float(xi) for xi in x]
    
    
    def predict(self, test):
        y_hat = [0] * len(test)
        for i, x in enumerate(test):
            sigma = self.calculate_sigma(x[1:])
            y_hat[i] = 1 if sigma > 0.5 else 0
        return y_hat

    
    def accuracy(self, test, predict_y):
        
        n = len(test[0])
        correct = 0
        for i, t in enumerate(test):
            if int(float(t[n-1])) == int(predict_y[i]):
                correct += 1
        print('Accuracy: ', correct/len(test))
            


data = []
with open('features.csv') as f:
    file_read = csv.reader(f)
    data = list(file_read)
    
train, test = get_train_test_data(data)

lr = LogisticRegression(0.1)
lr.fit(train)


predict_y = lr.predict(test)
lr.accuracy(test, predict_y)
