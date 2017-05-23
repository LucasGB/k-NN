#!/usr/bin/python
from sys import argv
import numpy as np
from itertools import groupby

def extract1(trainningFile):
	with open(trainningFile) as file:
		
		# Loads matrix from file
		trainning_characteristics = np.loadtxt(file)

		n_rows = len(trainning_characteristics)
		n_columns = len(trainning_characteristics[0])

		# Retrieves a list containing all the class numbers from file
		classes = []
		for i in range(n_rows):
			classes.append(trainning_characteristics[i][n_columns-1])

		used = set()
		# Creates a distinct label list
		distinctClasses = [x for x in classes if x not in used and (used.add(x) or True)]
		# Returns a list containing the sum for each distinct label occourrences
		# Gets the occurence of each class inside de file
		occurence = [classes.count(distinctClasses[x]) for x in range(len(distinctClasses))]

		print(occurence)


def extract(trainningFile):
	with open(trainningFile) as file:
		trainning_characteristics = np.loadtxt(file)
	return trainning_characteristics

def calculateDistance(trainning_characteristics, testFile, k):
	
	n_columns = len(trainning_characteristics[0])

	with open(testFile) as file:
		test_characteristics = np.loadtxt(testFile)

		distance_list = []
		k_closest_classes = []

		# Calculates the distance of the unknown instance for every neighbour
		for instance in test_characteristics:
			distance_list.append([np.linalg.norm(np.array(instance)-i) for i in trainning_characteristics])			

		for i in range(len(distance_list)):
			# Appends the k nearest classes
			aux = np.array(distance_list[i])
			ind = np.argpartition(aux, k)
			k_closest_classes.append([[trainning_characteristics[i][n_columns-1]] for j in ind[:k]])
				
	return k_closest_classes

def majority_vote(L):
  return max(groupby(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]



if __name__ == '__main__':

	if(len(argv) == 4):
		trainning_characteristics = extract(argv[1])

		k_closest_classes = calculateDistance(trainning_characteristics, argv[2], int(argv[3]))

		print majority_vote(k_closest_classes[0])




#	elif(len(argv) == 4 and argv[1] == 'centroid'):
#		extract1(argv[2])