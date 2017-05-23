#!/usr/bin/python
import sys
import numpy as np

#python knn.py Face\ Features/treino/features.txt Face\ Features/treino/labels.txt  Face\ Features/teste/features.txt Face\ Features/teste/labels.txt

def displayCentroids(centroid):
	# Prints resulting centroids list
	for i in range(len(centroid)):
		print()
		print(centroid[i])

def getOccurrences(fileName):
	file = open(fileName, 'r')

	# Removes the \n from each line of the file
	labels = [line.strip('\n') for line in file]
	used = set()
	# Creates a distinct label list
	distinctLabels = [x for x in labels if x not in used and (used.add(x) or True)]
	# Returns a list containing the sum for each distinct label occourrences
	return [labels.count(distinctLabels[x]) for x in range(len(distinctLabels))]	

# YUDI IS GAY
def start_trainning(features, labels):
	
	with open(features, 'r') as file:
		# Loads the matrix from file
		trainning_characteristics = np.loadtxt(file)
	
		occ = getOccurrences(labels)

		# Sets the variables start and end as delimiters and returns the corresponding submatrix of the interval.
		# Then, it proceeds to calculate the median of the tuples	
		centroid = []
		offset = 0
		for i in range(len(occ)):
			start = offset
			end = start + occ[i]						
			submatrix = np.array(trainning_characteristics)[[x for x in range(start, end)],:]			
			centroid.append([sum(x)/occ[i] for x in zip(*submatrix)])
			offset += occ[i]

		return centroid		
	

def classify(features, centroid):
	with open(features, 'r') as file:
		characteristics = np.loadtxt(file)

		# Calculates euclidean distance for every centroid
		dist = []
		classes = []
		for line in characteristics:			
			dist.append([np.linalg.norm(np.array(line)-i) for i in centroid])
			
		# Transforms list of lists into tuple and adds the lowest element(distance) in the list
		for i in range(len(dist)):
			aux = dist[i]
			classes.append(aux.index(min(aux)))
				
		return classes

def printResults(classifications, labelsFile):	
	correct_classifications = 0	
	
	with open(labelsFile, 'r') as file:
		labels = [line.strip('\n') for line in file]

		n_lines = len(labels)

		for i in range(n_lines):
			if(labels[i] == classifications[i]):
				correct_classifications += 1	

	print("Taxa de acertos: %f%% (%d/%d)" % ((float(correct_classifications)/n_lines)*100, correct_classifications, n_lines))

		

centroid = start_trainning(sys.argv[1], sys.argv[2])


# Maps centroids and classes it belongs to	
dict = {
			'0' : 'will_exp.', 
			'1' : 'tom_exp.', 
			'2' : 'stu_exp.', 
			'3' : 'ste_exp.', 
			'4' : 'sar_exp.', 
			'5' : 'pat_exp.', 
			'6' : 'mike_exp.', 
			'7' : 'lib_exp.', 
			'8' : 'john_exp.', 
			'9' : 'jer_', 
			'10' : 'ian_exp.', 
			'11' : 'glen_exp.', 
			'12' : 'den_exp.', 
			'13' : 'dav_exp.', 
			'14' : 'dah_exp_', 
			'15' : 'chr_exp.',
			'16' : 'ant_exp.',
			'17' : 'and_exp.', 			
}		

classes = classify(sys.argv[3], centroid)

classifications = []
for i in classes:	
	 classifications.append(dict[str(i)])

printResults(classifications, sys.argv[4])