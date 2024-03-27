import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from math import acos

def print_values(p, x,y,i):
    predictions = [item for sublist in p for item in sublist]
    ground = []
    for index in  x:
        ground.extend(y[index])
    cm = confusion_matrix(predictions, ground)
    precision = precision_score(ground, predictions, average='macro')
    recall = recall_score(ground, predictions, average='macro')
    accuracy = accuracy_score(ground, predictions)
    print('Fold ' + str(i))
    print("\tConfusion matrix: "+str(cm))
    print("\tPrecision: " + str(precision))
    print("\tRecall: " + str(recall))
    print("\tAccuracy: " + str(accuracy))
    return cm,precision,recall,accuracy

def plot_sample_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = points.reshape(83, 3)

    # Plot landmarks
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='brown', marker='o')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend(["Originals"])

    plt.show()


if __name__ == "__main__":
    #arg1 = TREE / RF / SVM 
    #arg2 = O / T / R
    #arg3 = ./FacialLandmarks
    input = [sys.argv[1], sys.argv[2], sys.argv[3]]

    
    print("---------------------running experiment start-----------------------")
    print("Data Classifier:", sys.argv[1], "\tData Type:", sys.argv[2])
    print("------------------------reading data start--------------------------")
    
    features = []
    classes = []
    subjects = []

    for subdir, dir, items in os.walk(input[2]):
        for item in items:
            label = subdir.split(os.path.sep)[-1]
            if item.endswith('.bnd'):
                with open((os.path.join(subdir, item)), 'r') as file:
                    data = file.readlines()
                    points = [list(map(float, line.split()[1:])) for line in data[:84]]
                    if input[1] == 'O' or input[1] == 'o': # for Original
                        points = np.array(points).flatten()
                    elif input[1] == 'T' or input[1] == 't': # for Translated
                        points = np.array(points)
                        points[:,0] -= (np.mean(points[:,0]))
                        points[:,1] -= (np.mean(points[:,1]))
                        points[:,2] -= (np.mean(points[:,2]))
                        points = points.flatten()
                    elif input[1] == 'RY' or input[1] == 'ry': # for Rotated Y
                        points = np.array(points)
                        pi = round(2*acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)

                        rotatedY = ((np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])).dot(points.T).T).flatten()
                        points =  rotatedY
                    elif input[1] == 'RX' or input[1] == 'rx': # for RotatedY
                        points = np.array(points)
                        pi = round(2*acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)

                        rotatedX = ((np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])).dot(points.T).T).flatten()
                        points = rotatedX
                    elif input[1] == 'RZ' or input[1] == 'rz': # for RotatedZ
                        points = np.array(points)
                        pi = round(2*acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)
                        rotatedZ = ((np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])).dot(points.T).T).flatten()
                        points = rotatedZ
                    
                    if label == 'Angry': val = 0
                    elif label == "Disgust" : val = 1
                    elif label == "Fear" : val = 2
                    elif label == "Happy" : val = 3
                    elif label == "Sad" : val = 4
                    elif label == "Surprise" : val = 5
                    # print("------------------------plotting data start-----------------------")
                    # plot_sample_points(points)
                    # print("-------------------------plotting data end------------------------")

                    features.append(points)
                    classes.append(val)
                    subjects.append(str(os.path.join(subdir, item)))

    print("-------------------------reading data end---------------------------")
    print("-------------------------evaluate data start------------------------")
    

    X = np.array(features)
    Y = np.array(classes)
    subjects = np.array(subjects)

    datatype = []
    if input[1] == 'RY' or input[1] == 'ry':
        datatype = [[X,"Rotated Y"]]
    elif input[1] == 'RZ' or input[1] == 'rz':
        datatype = [[X,"Rotated Z"]]
    elif input[1] == 'O' or input[1] == 'o':
        datatype = [[X,"Original"]]
    elif input[1] == 'T'  or input[1] == 't':
        datatype = [[X,"Translated"]]
    elif input[1] == 'RX'  or input[1] == 'rx':
        datatype = [[X,"Rotated X"]]

    if input[0] == 'TREE':
        classifier = DecisionTreeClassifier()

    elif input[0] == 'RF':
        classifier = RandomForestClassifier()

    elif input[0] == 'SVM':
        classifier = svm.LinearSVC(dual=False)

    
    for c, d in datatype:
        n = 10
        CM = precision = recall = accuracy = []
        groups = GroupKFold(n_splits = n)
        groups.get_n_splits(c, Y, subjects)

        for i in range(n):
            for index, (train_index, test_index) in enumerate(groups.split(c, Y, subjects)):
                if i == index:
                    Xtrain = c[train_index]
                    Xtest = c[test_index]
                    Ytrain = Y[train_index]
                    Ytest = Y[test_index]
                    classifier.fit(Xtrain, Ytrain)
                    p = classifier.predict(Xtest)
                    currcm , currprec, currrecall, curraccuracy = print_values([p],[test_index],Y,index+1)
                    CM.append(currcm)
                    precision.append(currprec)
                    recall.append(currrecall)
                    accuracy.append(curraccuracy)

                    gTrain = subjects[train_index]
                    gTest = subjects[test_index]
                    new = set(gTest) - set(gTrain)
                    newI = [i for i, subj in enumerate(subjects) if subj in new]
                    Xtrain = np.concatenate((Xtrain, X[newI]))
                    Ytrain = np.concatenate((Ytrain, Y[newI]))
                    
    print("-------------------------evaluate data end------------------------")
   