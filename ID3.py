import numpy
import csv
import pandas


class ID3:

    def fit_predict(self, train, test):

        with open('train.csv') as csvfile:
            data = list(csv.reader(csvfile))
            last = data[len(data) - 1][1]

