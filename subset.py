import csv
import gzip
import numpy as np
import time

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'


with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    # Load the data.
    totalTrain = []
    totalTest = []
    i = 0
    for row in train_csv:
        #smiles   = row[0]
        #features = np.array([float(x) for x in row[1:257]])
        #gap      = float(row[257])
        
        if(i < 100000):
        	totalTrain.append(row)

        elif i < 70000:
        	totalTest.append(row)
        elif i >= 70000:
        	break
        else:
        	print "FATAL ERROR!!!"
        i += 1

with open('smallTrain.csv', 'wb') as tiny:
	writer = csv.writer(tiny)
	for row in totalTrain:
		writer.writerow(row)


'''
print len(totalTest)

with open('tinyTest.csv', 'wb') as tinyTest:
	writer = csv.writer(tinyTest)
	for row in totalTest:
		writer.writerow(row)
'''


print "completely done!"


