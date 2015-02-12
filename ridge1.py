import csv
import numpy as np
import pylab as pl
import time
# from sklearn.linear_model import Ridge
# import sklearn
# from sklearn.decomposition import PCA
# from sklearn.svm import SVR

train_filename = 'tinyTrain.csv'
test_filename  = 'tinyTest.csv'

train_data = []

def RMSE(pred, gap_vector):
	summedError = []
	for i in xrange(len(pred)):
		summedError.append((pred[i] - gap_vector[i])**2.0)
	return np.sqrt((1.0/len(pred)*sum(summedError)))



with open(train_filename, 'r') as csv_fh:
	# Reads the csv file line by line
    reader = csv.reader(csv_fh)
	# Advances the reader past the header
    next(reader, None)
	# Holds the features of the current row
    features = np.zeros(shape=(1,256));
	# Holds a dynamic array of HOMO-LUMO gaps
    gap_array = []
	# list of features-gap pairs
    features_stack = []
	# Process the csv line by line
    for row in reader:
		# Grab the feature array for this molecule
        features = np.array([float(x) for x in row[1:257]])
		# Append the feature array for this molecule to the larger
		# feature matrix
    	features_stack.append(features)
		# Grab the information on HOMO-LUMO gap and append that info
		# to the vector of gaps
        gap = float(row[257])
        gap_array.append(gap)
		# Add to teh feature-gap pairing list
        train_data.append({'features': features, 'gap': gap})

# Create feature matrix and HOMO-LUMO gaps matrix
features_matrix = np.vstack(features_stack)
gap_vector = np.array(gap_array)

# Multiply the feature matrix by its inverse
square_features = np.dot(features_matrix.T, features_matrix)

# Define the lamdba penalty scalar and the "ridge" matrix
lambda_penalty = 4
ridge_identity = lambda_penalty * np.identity(256)

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(square_features + lambda_penalty * ridge_identity, 
					np.dot(features_matrix.T, gap_vector))

# Sanity checks to make sure that matrices, vectors, and dimensions
# Look correct for the relevant variables in the problem

print "Dimensions of the feature matrix: ", features_matrix.shape
print "Features matrix looks like: " 
print features_matrix

print "Dimension of the gap vector: ", gap_vector.shape
print "Gap vector looks like: ", gap_vector

print "Feature matrix times its inverse has shape: ", square_features.shape
print "Feature matrix times its transpose looks like: "
print square_features

print "Lambda penalty: ", lambda_penalty
print "Ridge matrix has shape: ", ridge_identity.shape
print "Ridge matrix looks like: "
print ridge_identity

print "Regression weights vector has shape: ", w.shape
print "Regression weights look like: "
print w

csv_fh.close()


with open(test_filename, 'r') as csv_test:
	# This will hold the actual HOMO-LUMO gap values
	YTest = []
	# This will hold the features of each row in test set
	features_arrayTest = []
	# CSV scanner
	reader = csv.reader(csv_test)
	# Advances the reader past the header
	next(reader, None)
	# Read through each row in the CSV
	for row in reader:
		# Pull out the gap info for the HUMO-LUMO gap
		gapTest = float(row[257])
		# Append the gap info to the gap list
		YTest.append(gapTest)
		# Append the feature info to the feature array list
		features_arrayTest.append(np.array([float(x) for x in row[1:257]]))		

# Create a test matrix and YTest --> the actual HOMO-LUMO gaps
test_matrix = np.vstack(features_arrayTest)
actual_gaps_vector = np.array(YTest)

# Use the solution weights to predict the HOMO-LUMO gaps
predicted_gaps_vector = np.dot(test_matrix, w)

# Sanity check to make sure the matrices and arrays look fine

print "Test matrix has shape: ", test_matrix.shape
print "Test matrix looks like: "
print test_matrix

print "Actual HOMO-LUMO gaps vector has shape: ", actual_gaps_vector.shape
print "HOMO-LUMO actual values look like: ", 
print actual_gaps_vector

print "Predicted HOMO-LUMO gaps vector has shape: ", predicted_gaps_vector.shape
print "HOMO-LUMO predicted values look like: ", 
print predicted_gaps_vector

# Calculate the error of the predicted versus the actual values
rmse = RMSE(predicted_gaps_vector, actual_gaps_vector)
print "RMSE with ridge method: ", rmse


'''
# pca = PCA(n_components=100)

# P = pca.fit_transform(features_array)
# features_array = P

# P2 = pca.fit_transform(features_arrayTest)
# features_arrayTest = P2

features_array = np.array(features_array)
gap_vector = np.array(gap_vector)

features_arrayTest = np.array(features_arrayTest)
YTest = np.array(YTest)
'''

'''
best_ridge_Error = 99999
best_ridge_alpha = 0.0
for aR in np.arange(0, 2.1, 0.1):
	ridgeRegression = Ridge(alpha=aR)
	ridgeRegression.fit(features_array, gap_vector)
	predictionsRidge = ridgeRegression.decision_function(features_arrayTest)
	error = RMSE(predictionsRidge, YTest)
	if error < best_ridge_Error:
		best_ridge_Error = error
		best_ridge_alpha = aR

print best_ridge_alpha
print "the best ridge error was : ", best_ridge_Error



lassoRegression = sklearn.linear_model.Lasso(alpha=1.0)
lassoRegression.fit(features_array, gap_vector)
predictionsLasso = lassoRegression.decision_function(features_arrayTest)

elasticRegression = sklearn.linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticRegression.fit(features_array, gap_vector)
predictionsElastic = elasticRegression.decision_function(features_arrayTest)

print "the RMSE for ridge is : ", RMSE(predictionsRidge, YTest)
print "the RMSE for lasso is : ", RMSE(predictionsLasso, YTest)
print "the RMSE for elasticRegression is : ", RMSE(predictionsElastic, YTest)

'''

'''
svRegression = SVR(C=5.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 0 is : ",  RMSE(predictionsSV, YTest)

svRegression = SVR(C=100.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 1 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(C=1000.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 2 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(C=10000.0, epsilon=0.01)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 3 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(C=10000.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 4 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=1.0, epsilon=0.2)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 1 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=1.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 2 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=10.0, epsilon=0.01)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 3 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=10000.0, epsilon=0.1)
svRegression.fit(features_array, gap_vector)
predictionsSV = svRegression.decision_function(features_arrayTest)
print "number 4 is : ", RMSE(predictionsSV, YTest)


'''






