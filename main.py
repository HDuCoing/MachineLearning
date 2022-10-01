import som
import mlp
import numpy as np
import matplotlib
import pylab as pl
# Load data into numpy arrays
x_data = np.loadtxt('train/X_train.txt', dtype=float)
y_data = np.loadtxt('train/y_train.txt', dtype=int)
subjects = np.loadtxt('train/subject_train.txt', dtype=int)
#x_data = np.array(x_data)
#y_data = np.array(y_data)
#subjects = np.array(subjects)

# Taken from fittest genes in assignment 1
fittestGenes = [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]


# Filter data using the list of 561 genes from GA's fittest genes.
def filterItems(filterMe, chromosome):
    filtered = []
    for y in range(len(filterMe)):
        item = []
        for pos, x in enumerate(chromosome):
            if x == 1:
                item.append(filterMe[y][pos])
        filtered.append(item)
    filtered = np.array(filtered)
    return filtered
# These are the filtered training and test sets.
trainFiltered = filterItems(x_data, fittestGenes)
# Create the self-organising map and cluster the best features, then plot it.
def makesom(x,y):
    # Self organizing map constructor.
    net = som.som(x, y, trainFiltered)
    net.somtrain(trainFiltered, 561)

    # Store best node for each train input
    best = np.zeros(np.shape(trainFiltered)[0], dtype=int)
    for i in range(np.shape(trainFiltered)[0]):
        best[i], activation = net.somfwd(trainFiltered[i, :])

    # plot SOM train data
    pl.plot(net.map[0, :], net.map[1, :], 'k.', ms=15)
    where = np.where(y_data == 1)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'rs', ms=15)
    where = np.where(y_data == 2)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'gv', ms=15)
    where = np.where(y_data == 3)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'b^', ms=15)
    where = np.where(y_data == 4)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'c>', ms=15)
    where = np.where(y_data == 5)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'y<', ms=15)
    where = np.where(y_data == 6)
    pl.plot(net.map[0, best[where]], net.map[1, best[where]], 'mo', ms=15)
    pl.axis([-0.1, 1.1, -0.1, 1.1])
    pl.axis('off')
    pl.tight_layout()
    pl.show()
    return best


def countoverlaps(target,best):
    # Find places where the same neuron represents different classes
    i1 = np.where(target==1)
    nodes0 = np.unique(best[i1])
    i2 = np.where(target==2)
    nodes1 = np.unique(best[i2])
    i3 = np.where(target==3)
    nodes2 = np.unique(best[i3])
    i4 = np.where(target==4)
    nodes3 = np.unique(best[i4])

    i5 = np.where(target==5)
    nodes4 = np.unique(best[i4])
    i6 = np.where(target==6)
    nodes5 = np.unique(best[i4])

    doubles01 = np.in1d(nodes0,nodes1,assume_unique=True)
    doubles02 = np.in1d(nodes0,nodes2,assume_unique=True)
    doubles12 = np.in1d(nodes1,nodes2,assume_unique=True)
    doubles23 = np.in1d(nodes2, nodes3, assume_unique=True)
    doubles34 = np.in1d(nodes3, nodes4, assume_unique=True)
    doubles45 = np.in1d(nodes4, nodes5, assume_unique=True)

    return len(nodes0[doubles01]) + len(nodes0[doubles02]) + \
           len(nodes1[doubles12]) + \
           len(nodes2[doubles23]) + len(nodes3[doubles34]) + \
           len(nodes4[doubles45]) + len(nodes5[doubles45])

# See if a user is recognized based on their walking.
def recognizeUser():
    # obtain users and features that == 1 (walking)
    xWalking = np.array(x_data[np.where(y_data == 1)])
    ySubjects = np.array(subjects[np.where(y_data == 1)])
    print("walking", xWalking)
    print("subjects", ySubjects)
    # x data sets
    train = xWalking[0:400]
    valid = xWalking[400:800]
    test = xWalking[800:1200]
    # y data sets
    ytrain = ySubjects[0:400]
    yvalid = ySubjects[400:800]
    ytest = ySubjects[800:1200]

    # create a matrix in the length of the y data set, and for the amount of users (subjects)
    # Train subject set
    trainSubject = np.zeros((len(ytrain), 30))
    for i in range(len(ytrain)):
        trainSubject[i, ytrain[i]-1] = 1
    # Valid subject set
    validSubject = np.zeros((len(yvalid), 30))
    for i in range(len(yvalid)):
        validSubject[i, yvalid[i]-1] = 1
    # Test subject set
    testSubject = np.zeros((len(ytest), 30))
    for i in range(len(ytest)):
        testSubject[i, ytest[i]-1] = 1

    for i in [1, 5, 10, 15, 20]:
        print("Amount of Hidden Layers: ", i)
        network = mlp.mlp(train, trainSubject, nhidden=5, outtype='softmax')
        network.earlystopping(train, trainSubject, valid, validSubject, eta=0.25)
        network.confmat(test, testSubject)

def callSOM():
    # Track how many neurons are overlapping.io
    score = np.zeros((6,30))
    # current score
    count = 0
    for x in [30, 30]:
        for y in [30, 30]:
            best = makesom(x, y)
            Noverlaps = countoverlaps(y_data, best)
            # So a possible score is:
            score[count] = x * y + 10 * Noverlaps
            count += 1

    print(score)
    # Now pick the best
    print(np.argmin(score))

#callSOM()
recognizeUser()
