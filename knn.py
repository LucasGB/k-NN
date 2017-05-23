#!/usr/bin/python
from sys import argv
from datetime import datetime
import numpy as np
from itertools import groupby
from sklearn.metrics import confusion_matrix
from scipy import stats
from scipy import spatial

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


def separateFiles(trainningFile, testFile):
	with open(trainningFile, 'r') as file:
		data = np.loadtxt(file)
		labels = data[:, [len(data[0])-1]]
		characteristics = np.delete(data, -1, 1)

		trainningLabelsFile = trainningFile+'_labels.txt'
		trainningCharacteristicsFile = trainningFile+'_characteristics.txt'

		np.savetxt(trainningLabelsFile, labels)
		np.savetxt(trainningCharacteristicsFile, characteristics)

	with open(testFile, 'r') as file:
		data = np.loadtxt(file)
		labels = data[:, [len(data[0])-1]]
		characteristics = np.delete(data, -1, 1)

		testLabelsFile = testFile+'_labels.txt'
		testCharacteristicsFile = testFile+'_characteristics.txt'

		np.savetxt(testLabelsFile, labels)
		np.savetxt(testCharacteristicsFile, characteristics)

def extract(fileName):
	with open(fileName) as file:
		data = np.loadtxt(file)
	return data

def euclideanDistance(trainning_characteristics, trainning_labels, testFile):

	testLabelsFile = testFile+'_labels.txt'
	testCharacteristicsFile = testFile+'_characteristics.txt'

	test_labels = extract(testLabelsFile)
	test_characteristics = extract(testCharacteristicsFile)

	distance_list = []

	# Calculates the distance of the unknown instance for every neighbour
	for instance in test_characteristics:
		distance_list.append([np.linalg.norm(np.array(instance)-i) for i in trainning_characteristics])

	return test_labels, distance_list

def manhattanDistance(trainning_characteristics, trainning_labels, testFile):
	testLabelsFile = testFile+'_labels.txt'
	testCharacteristicsFile = testFile+'_characteristics.txt'

	test_labels = extract(testLabelsFile)
	test_characteristics = extract(testCharacteristicsFile)

	distance_list = []

	# Calculates the distance of the unknown instance for every neighbour
	for instance in test_characteristics:
		#distance_list.append([np.linalg.norm(np.array(instance)-i) for i in trainning_characteristics])
		#ditsance_list.append(   [    [sum(abs(instance[j]-characteristics[j])) for characteristics in trainning_characteristics]]   )
		#distance_list.append(sum(abs(instance-i) for instance, i in zip(test_characteristics, trainning_characteristics)))
		distance_list.append([spatial.distance.cityblock(instance, i) for i in trainning_characteristics])

	print len(distance_list)
	print len(distance_list[0])
	return test_labels, distance_list

def get_k_closests(distance_list, k):
	k_closest_classes = []

	for i in range(len(distance_list)):
		# Appends the k nearest classes
		aux = np.array(distance_list[i])
		ind = np.argpartition(aux, k)
		k_closest_classes.append([trainning_labels[j] for j in ind[:k]])	

	return k_closest_classes

def classify(k_closest_classes):
	classified = []
	for i in k_closest_classes:
		classified.append(majority_vote(i))
	return classified

def majority_vote(L):
  return max(groupby(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]

def normalizeMinMax(trainning_characteristics):
 	normalized_list = []
 	min_list = []
 	max_list = []
	for characteristics in trainning_characteristics:		
		min_list.append(characteristics.tolist().index(min((characteristics))))
		max_list.append(characteristics.tolist().index(max((characteristics))))

	[normalized_list.append([((characteristics[i] - min_list[i]) / float(max_list[i] - min_list[i])) for i in range(len((characteristics)))]) for characteristics in trainning_characteristics]
	return normalized_list

def normalize_z_score(trainning_characteristics):
	normalized_list = []
	[[normalized_list.append(stats.zscore(characteristics))] for characteristics in trainning_characteristics]
	return normalized_list

def buildConfusionMatrix(classified, trainning_labels, test_labels):
	n_rows = len(trainning_labels)
	n_columns = len(trainning_labels)
	
	# Creates a list with unique labels names from the trainning labels
	used = set()
	distinctLabels = [x for x in trainning_labels if x not in used and (used.add(x) or True)]
	confusionMatrix = confusion_matrix(test_labels, classified)
	print 'Linhas -> classes\nColunas -> classificadas'
	print confusionMatrix

if __name__ == '__main__':
	startTime = datetime.now()

	if(len(argv) == 6 and argv[4] == 'euclidean' and argv[5] == 'minmax'):
		separateFiles(argv[1], argv[2])

		trainningLabelsFile = argv[1]+'_labels.txt'
		trainningCharacteristicsFile = argv[1]+'_characteristics.txt'	

		trainning_labels = extract(trainningLabelsFile)
		trainning_characteristics = extract(trainningCharacteristicsFile)		

		normalized_characteristics = normalizeMinMax(trainning_characteristics)

		test_labels, distance_list = euclideanDistance(normalized_characteristics, trainning_labels, argv[2])

		k_closest_classes = get_k_closests(distance_list, int(argv[3]))

		classified = classify(k_closest_classes)		

		buildConfusionMatrix(classified, trainning_labels, test_labels)


	elif(len(argv) == 6 and argv[4] == 'euclidean' and argv[5] == 'zscore'):
		separateFiles(argv[1], argv[2])

		trainningLabelsFile = argv[1]+'_labels.txt'
		trainningCharacteristicsFile = argv[1]+'_characteristics.txt'	

		trainning_labels = extract(trainningLabelsFile)
		trainning_characteristics = extract(trainningCharacteristicsFile)		

		normalized_characteristics = normalize_z_score(trainning_characteristics)

		test_labels, distance_list = euclideanDistance(normalized_characteristics, trainning_labels, argv[2])

		k_closest_classes = get_k_closests(distance_list, int(argv[3]))

		classified = classify(k_closest_classes)		

		buildConfusionMatrix(classified, trainning_labels, test_labels)

	elif(len(argv) == 6 and argv[4] == 'manhattan' and argv[5] == 'minmax'):
		separateFiles(argv[1], argv[2])

		trainningLabelsFile = argv[1]+'_labels.txt'
		trainningCharacteristicsFile = argv[1]+'_characteristics.txt'	

		trainning_labels = extract(trainningLabelsFile)
		trainning_characteristics = extract(trainningCharacteristicsFile)		

		normalized_characteristics = normalizeMinMax(trainning_characteristics)

		test_labels, distance_list = manhattanDistance(normalized_characteristics, trainning_labels, argv[2])

		k_closest_classes = get_k_closests(distance_list, int(argv[3]))

		classified = classify(k_closest_classes)		

		buildConfusionMatrix(classified, trainning_labels, test_labels)


	elif(len(argv) == 6 and argv[4] == 'manhattan' and argv[5] == 'zscore'):
		separateFiles(argv[1], argv[2])

		trainningLabelsFile = argv[1]+'_labels.txt'
		trainningCharacteristicsFile = argv[1]+'_characteristics.txt'	

		trainning_labels = extract(trainningLabelsFile)
		trainning_characteristics = extract(trainningCharacteristicsFile)		

		normalized_characteristics = normalize_z_score(trainning_characteristics)

		test_labels, distance_list = manhattanDistance(normalized_characteristics, trainning_labels, argv[2])

		k_closest_classes = get_k_closests(distance_list, int(argv[3]))

		classified = classify(k_closest_classes)		

		buildConfusionMatrix(classified, trainning_labels, test_labels)

	print datetime.now() - startTime,'segundos'