import csv
import numpy as np
import pylab as pl
import time
from sklearn.linear_model import Ridge
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVR

train_filename = 'tinyTrain.csv'
test_filename  = 'tinyTest.csv'

train_data = []

def RMSE(pred, Y):
	summedError = []
	for i in xrange(len(pred)):
		summedError.append((pred[i] - Y[i])**2.0)
	return np.sqrt((1.0/len(pred)*sum(summedError)))



with open(train_filename, 'r') as csv_fh:

    reader = csv.reader(csv_fh)

    next(reader, None)
    features = []
    Y = []
    feats = []
    for row in reader:
    	feats.append(np.array([float(x) for x in row[1:257]]))
        features = np.array([float(x) for x in row[1:257]])
        gap = float(row[257])
        Y.append(gap)
        train_data.append({'features': features,
        					'gap': gap})

csv_fh.close()

with open(test_filename, 'r') as csv_test:

	YTest = []
	featsTest = []

	reader = csv.reader(csv_test)

	time.sleep(10)
	for row in reader:
		gapTest = float(row[257])
		YTest.append(gapTest)
		featsTest.append(np.array([float(x) for x in row[1:257]]))		


# pca = PCA(n_components=100)

# P = pca.fit_transform(feats)
# feats = P

# P2 = pca.fit_transform(featsTest)
# featsTest = P2

feats = np.array(feats)
Y = np.array(Y)

featsTest = np.array(featsTest)
YTest = np.array(YTest)

'''
best_ridge_Error = 99999
best_ridge_alpha = 0.0
for aR in np.arange(0, 2.1, 0.1):
	ridgeRegression = Ridge(alpha=aR)
	ridgeRegression.fit(feats, Y)
	predictionsRidge = ridgeRegression.decision_function(featsTest)
	error = RMSE(predictionsRidge, YTest)
	if error < best_ridge_Error:
		best_ridge_Error = error
		best_ridge_alpha = aR

print best_ridge_alpha
print "the best ridge error was : ", best_ridge_Error



lassoRegression = sklearn.linear_model.Lasso(alpha=1.0)
lassoRegression.fit(feats, Y)
predictionsLasso = lassoRegression.decision_function(featsTest)

elasticRegression = sklearn.linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticRegression.fit(feats, Y)
predictionsElastic = elasticRegression.decision_function(featsTest)

print "the RMSE for ridge is : ", RMSE(predictionsRidge, YTest)
print "the RMSE for lasso is : ", RMSE(predictionsLasso, YTest)
print "the RMSE for elasticRegression is : ", RMSE(predictionsElastic, YTest)

'''

'''
svRegression = SVR(C=5.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 0 is : ",  RMSE(predictionsSV, YTest)

svRegression = SVR(C=100.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 1 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(C=1000.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 2 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(C=10000.0, epsilon=0.01)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 3 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(C=10000.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 4 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=1.0, epsilon=0.2)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 1 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=1.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 2 is : ",  RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=10.0, epsilon=0.01)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 3 is : ", RMSE(predictionsSV, YTest)


svRegression = SVR(kernel='poly', C=10000.0, epsilon=0.1)
svRegression.fit(feats, Y)
predictionsSV = svRegression.decision_function(featsTest)
print "number 4 is : ", RMSE(predictionsSV, YTest)


'''






