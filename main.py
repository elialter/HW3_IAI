# This is a sample Python script.
from ID3 import *
import numpy
import csv
import pandas
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    id3 = ID3()

    with open('train.csv') as csvfile:
        data = list(csv.reader(csvfile))
    array = numpy.array(data)
    data.sort(key=lambda x: x[3])
    for i in data:
        print(i)


    id3.fit_predict(array, array)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
