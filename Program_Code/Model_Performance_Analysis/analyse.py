from __future__ import division
import csv
import os
import glob
import numpy as np


fileName = ""
fileData = []
accuracy = 0

# [row][column]
# [[TP, FP],[FN, TN]]
mocAD = [[0,0],[0,0]]
mocMCI = [[0,0],[0,0]]
mocNL = [[0,0],[0,0]]
mocAVG = [[0,0],[0,0]]

# [row][column]
#	    NL   MCI   AD
#	NL
#	MCI
#	AD
confMatr = [[0,0,0],[0,0,0],[0,0,0]]


# So that divide by 0 doesn't exit the program
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


def getFileName():
	global fileName
	for file in glob.glob("*.csv"):
		fileName = file


def readFile():
	global fileData
	with open(fileName, "rb") as file:
		reader = csv.reader(file)
		data = list(reader)
		for row in data:
			rowData = row[0].split(";")
			if rowData[0] != "Actual":
				fileData.append(rowData)

# Calculates % correct
def basicMetrics():
	count = 0
	for row in fileData:
		if row[0] == row[1]:
			count += 1
	global accuracy
	accuracy = (count / len(fileData)) * 100


# Create the tables of confusion
def createMOCs():
	global mocAD, mocMCI, mocNL, mocAVG
	for row in fileData:
		actual = row[0]
		predicted = row[1]

		# AD MOC
		if actual == "AD":
			if predicted == "AD":
				mocAD[0][0] += 1
			else:
				mocAD[1][0] += 1
		else:
			if predicted == "AD":
				mocAD[0][1] += 1
			else:
				mocAD[1][1] += 1
		# MCI MOC
		if actual == "MCI":
			if predicted == "MCI":
				mocMCI[0][0] += 1
			else:
				mocMCI[1][0] += 1
		else:
			if predicted == "MCI":
				mocMCI[0][1] += 1
			else:
				mocMCI[1][1] += 1
		# NL MOC
		if actual == "NL":
			if predicted == "NL":
				mocNL[0][0] += 1
			else:
				mocNL[1][0] += 1
		else:
			if predicted == "NL":
				mocNL[0][1] += 1
			else:
				mocNL[1][1] += 1
	# Average the MOC's
	mocAVG[0][0] = (mocAD[0][0] + mocMCI[0][0] + mocNL[1][1]) / 3
	mocAVG[0][1] = (mocAD[0][1] + mocMCI[0][1] + mocNL[0][1]) / 3
	mocAVG[1][0] = (mocAD[1][0] + mocMCI[1][0] + mocNL[1][0]) / 3
	mocAVG[1][1] = (mocAD[1][1] + mocMCI[1][1] + mocNL[0][0]) / 3


# Create the confusion matrix
def createConfMatr():
	global confMatr
	for row in fileData:
		actual = row[0]
		predicted = row[1]
		if actual == "AD":
			if predicted == "AD":
				confMatr[0][0] += 1
			elif predicted == "MCI":
				confMatr[1][0] += 1
			elif predicted == "NL":
				confMatr[2][0] += 1
		elif actual == "MCI":
			if predicted == "AD":
				confMatr[0][1] += 1
			elif predicted == "MCI":
				confMatr[1][1] += 1
			elif predicted == "NL":
				confMatr[2][1] += 1
		elif actual == "NL":
			if predicted == "AD":
				confMatr[0][2] += 1
			elif predicted == "MCI":
				confMatr[1][2] += 1
			elif predicted == "NL":
				confMatr[2][2] += 1


# Calculate different metrics
def details(moc):
	prevalence = safe_div((moc[0][0] + moc[0][1]), len(fileData))
	accuracy = safe_div((moc[0][0] + moc[1][1]), len(fileData))
	precision = safe_div(moc[0][0], (moc[0][0] + moc[0][1]))
	sensitivity = safe_div(moc[0][0], (moc[0][0] + moc[1][0]))
	specificity = safe_div(moc[1][1], (moc[0][1] + moc[1][1]))
	f1 = safe_div(2, ((safe_div(1,sensitivity)) + (safe_div(1,precision))))

	print "Prevalence: ", prevalence
	print "Accuracy: ", accuracy
	print "Precision: ", precision
	print "Sensitivity: ", sensitivity
	print "Specificity: ", specificity
	print "F1 score: ", f1
	print "\n"


def printEverything():
	# Accuracy
	print "Accuracy: ", accuracy, "%\n"
	# MOC's
	print "AD MOC\n",np.matrix(mocAD)
	details(mocAD)
	print "MCI MOC\n",np.matrix(mocMCI)
	details(mocMCI)
	print "NL MOC\n",np.matrix(mocNL)
	details(mocNL)
	print "Avg. MOC\n",np.matrix(mocAVG)
	details(mocAVG)
	# Conf matrix
	print "Confusin Matrix\n", np.matrix(confMatr),"\n"


def main():
	getFileName()
	readFile()
	basicMetrics()
	createMOCs()
	createConfMatr()
	printEverything()


if __name__ == "__main__":
    main()
