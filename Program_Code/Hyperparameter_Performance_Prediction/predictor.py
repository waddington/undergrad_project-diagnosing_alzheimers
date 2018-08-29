import pandas as pd
import quandl as Quandl
import math, datetime, time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import csv

# Used to store the data
xm = list()
ym = list()

# Read the file data
def get_data(filename):
    global xm, ym
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        for row in csvFileReader:
            rowNmr = float(''.join(str(e) for e in row).strip('"'))
            ym.append(rowNmr)
    xm = range(0, len(ym), 1)
    return


get_data("db7.csv")

# Making the list fit the required shape for the regression model
xm = np.reshape(xm, (len(ym), 1))

# Creating and fitting a model
model = LinearRegression()
model.fit(xm, ym)

# Line coefficient
coef = float(model.coef_)
print('Coef: {:.20f}'.format(coef))

# Plotting the information
y_pred = model.predict(xm)

plt.scatter(xm, ym)
plt.plot(xm, y_pred, 'r-')
plt.ylabel("MSE");
plt.xlabel("Training Example");
plt.show()
