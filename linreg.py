import csv
import numpy as np
import pylab as pl
import time

csv_filename = 'motorcycle.csv'

times  = []
forces = []
with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        times.append(float(row[0]))
        forces.append(float(row[1]))

# Turn the data into numpy arrays.
times  = np.array(times)
forces = np.array(forces)

# Plot the data.
# pl.plot(times, forces, 'o')
# pl.show()

# Create the simplest basis, with just the time and an offset.

def function(x, j):
    #return np.e**(-((x - 10*j)/5)**2)
    #return np.e**(-((x - 10*j)/10)**2)
    return np.e**(-((x - 10*j)/25)**2)
    #return x**j
    #return np.sin(x/j)
     

j_max = 6

a = ()
for x in times: 
    point = [1]
    for j in xrange(0, j_max + 1):
        #s = np.e**(-((x - 10*j)/5)**2)
        s = function(x, j)
        point.append(s)
    p = np.array(point)
    a = a + (p, )


X = np.vstack(a)
print "The shape of the first X is : ", X.shape

#X = np.vstack((np.ones(times.shape), times)).T

#print "the shape of the second X is : ", X.shape
#time.sleep(5)

# Nothing fancy for outputs.
Y = forces

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))


# Compute the regression line on a grid of inputs.
grid_times = np.linspace(0, 60, 200)
#grid_X     = np.vstack((np.ones(grid_times.shape), grid_times))

gridStuff = ()
for x in grid_times: 
    point = [1]
    for j in xrange(0, j_max + 1):
        s = function(x, j)
        point.append(s)
    #print point
    p = np.array(point)
    print "the shape of p is : ", p.shape
    #time.sleep(10)
    gridStuff = gridStuff + (p, )

grid_X = np.vstack(gridStuff)

print "X.T looks like", grid_X.shape 
print "and w looks like : ", w.shape
grid_Yhat  = np.dot(grid_X, w)

# Plot the data and the regression line.

print "grid_Yhat is : ", grid_Yhat.shape
print "grid_times is : ", grid_times.shape

pl.plot(times, forces, 'o',
        grid_times, grid_Yhat, '-')
pl.show()



