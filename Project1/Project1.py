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
    # Flatten the list of predictions. 'p' is expected to be a list of lists, where each sublist contains predictions for a fold.
    predictions = [item for sublist in p for item in sublist]
    ground = []
    for index in  x:
    # 'x' is expected to contain indices pointing to the elements in 'y' that belong to the current fold.
        ground.extend(y[index])
    #  Computing all the required values using predefined functions from sklearn
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
    fig = plt.figure() # Create a new figure for plotting.
    ax = fig.add_subplot(111, projection='3d') # Add a 3D subplot to the figure. '111' means 1x1 grid.
    points = points.reshape(83, 3) # Reshape the 'points' array to have 83 rows and 3 columns.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='brown', marker='o')# Scatter plot the points. 
    # Set labels for each axis.
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend(["Originals"]) # Legen for plot Change this to what we are plotting.
    plt.show() # Display the plot.


if __name__ == "__main__":
    #arg1 = TREE / RF / SVM 
    #arg2 = O / T / R
    #arg3 = ./FacialLandmarks
    input = [sys.argv[1], sys.argv[2], sys.argv[3]]

    print("---------------------running experiment start-----------------------")
    print("Data Classifier:", sys.argv[1], "\tData Type:", sys.argv[2])
    print("------------------------reading data start--------------------------")
    
    
    # Initialize lists to hold feature vectors, class labels, and subject identifiers.
    features = []
    classes = []
    subjects = []

    # Walk through the directory structure starting from a given root directory.
    for subdir, dirs, items in os.walk(input[2]):
        # Iterate over each file in the current directory.
        for item in items:
            # Extract the class label from the directory name.
            label = subdir.split(os.path.sep)[-1]
            # Process files that end with '.bnd'.
            if item.endswith('.bnd'):
                # Open and read the content of the file.
                with open(os.path.join(subdir, item), 'r') as file:
                    data = file.readlines()
                    # Extract and convert the first 84 lines of data into numerical format.
                    points = [list(map(float, line.split()[1:])) for line in data[:84]]
                    
                    # Apply transformations based on the input command.
                    # Flatten the points if the original format is requested.
                    if input[1] in ['O', 'o']:
                        points = np.array(points).flatten()
                    # Translate the points so their mean is zero and flatten them.
                    elif input[1] in ['T', 't']:
                        points = np.array(points)
                        points[:,0] -= np.mean(points[:,0])
                        points[:,1] -= np.mean(points[:,1])
                        points[:,2] -= np.mean(points[:,2])
                        points = points.flatten()
                    # Rotate around the Y-axis and flatten the points.
                    elif input[1] in ['RY', 'ry']:
                        points = np.array(points)
                        pi = round(2*np.acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)
                        rotatedY = np.dot(np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]), points.T).T.flatten()
                        points = rotatedY
                    # Rotate around the X-axis and flatten the points.
                    elif input[1] in ['RX', 'rx']:
                        points = np.array(points)
                        pi = round(2*np.acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)
                        rotatedX = np.dot(np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]]), points.T).T.flatten()
                        points = rotatedX
                    # Rotate around the Z-axis and flatten the points.
                    elif input[1] in ['RZ', 'rz']:
                        points = np.array(points)
                        pi = round(2*np.acos(0.0), 3)
                        cos = np.cos(pi)
                        sin = np.sin(pi)
                        rotatedZ = np.dot(np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]), points.T).T.flatten()
                        points = rotatedZ
                    
                    # Convert class labels from string to numerical format.
                    if label == 'Angry': val = 0
                    elif label == "Disgust": val = 1
                    elif label == "Fear": val = 2
                    elif label == "Happy": val = 3
                    elif label == "Sad": val = 4
                    elif label == "Surprise": val = 5

                    # Append the processed data to the respective lists.
                    features.append(points)
                    classes.append(val)
                    subjects.append(os.path.join(subdir, item))
    print("-------------------------reading data end---------------------------")
    print("-------------------------evaluate data start------------------------")
    

    # Convert the lists of features and classes into NumPy arrays for easier manipulation.
    X = np.array(features)
    Y = np.array(classes)
    subjects = np.array(subjects)

    # Determine the type of data processing based on user input and create a list that contains both the features and their corresponding label.
    datatype = []
    if input[1] in ['RY', 'ry']:
        datatype = [[X, "Rotated Y"]]
    elif input[1] in ['RZ', 'rz']:
        datatype = [[X, "Rotated Z"]]
    elif input[1] in ['O', 'o']:
        datatype = [[X, "Original"]]
    elif input[1] in ['T', 't']:
        datatype = [[X, "Translated"]]
    elif input[1] in ['RX', 'rx']:
        datatype = [[X, "Rotated X"]]

    # Select the machine learning classifier based on user input.
    if input[0] == 'TREE':
        classifier = DecisionTreeClassifier()
    elif input[0] == 'RF':
        classifier = RandomForestClassifier()
    elif input[0] == 'SVM':
        classifier = svm.LinearSVC(dual=False)

    # Iterate through each data processing type (although in this case, it's likely to be just one type).
    for c, d in datatype:
        n = 10  # Number of folds in cross-validation.
        # Initialize lists to store the evaluation metrics across all folds.
        CM = precision = recall = accuracy = []
        groups = GroupKFold(n_splits=n)
        groups.get_n_splits(c, Y, subjects)  # Prepare the splits based on the groups (subjects).

        for i in range(n):
            for index, (train_index, test_index) in enumerate(groups.split(c, Y, subjects)):
                if i == index:
                    # Split the data into training and testing sets for the current fold.
                    Xtrain, Xtest = c[train_index], c[test_index]
                    Ytrain, Ytest = Y[train_index], Y[test_index]
                    
                    # Train the classifier and predict on the test set.
                    classifier.fit(Xtrain, Ytrain)
                    p = classifier.predict(Xtest)
                    
                    # Evaluate the model's performance and accumulate the metrics.
                    currcm, currprec, currrecall, curraccuracy = print_values([p], [test_index], Y, index + 1)
                    CM.append(currcm)
                    precision.append(currprec)
                    recall.append(currrecall)
                    accuracy.append(curraccuracy)

                    # Identify new subjects that are in the test set but not in the training set.
                    gTrain = subjects[train_index]
                    gTest = subjects[test_index]
                    new = set(gTest) - set(gTrain)
                    newI = [i for i, subj in enumerate(subjects) if subj in new]
                    
                    # Augment the training set with these new subjects.
                    Xtrain = np.concatenate((Xtrain, X[newI]))
                    Ytrain = np.concatenate((Ytrain, Y[newI]))
    print("-------------------------evaluate data end------------------------")
   