
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import os


def PrintEvalMetrics(pred, indices, y, classifier, data_type, fold_index):
    # manually merge predictions and testing labels from each of the folds to make confusion matrix
    finalPredictions = []
    groundTruth = []

    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    cm = confusion_matrix(finalPredictions, groundTruth)
    precision = precision_score(
        groundTruth, finalPredictions, average='macro')
    recall = recall_score(groundTruth, finalPredictions, average='macro')
    accuracy = accuracy_score(groundTruth, finalPredictions)
    # print(cm)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("Accuracy: ", accuracy)
    new_dir = 'Final_Outputs/'+classifier
    os.makedirs(new_dir, exist_ok=True)
    filename = classifier+"_"+data_type+".txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Fold ' + str(fold_index)+'\n')
            file.write("\tConfusion matrix: "+str(cm) + '\n')
            file.write("\tPrecision: " + str(precision) + '\n')
            file.write("\tRecall: " + str(recall) + '\n')
            file.write("\tAccuracy: " + str(accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)
    return cm, precision, recall, accuracy


def subject_independent_cross_validation(X, y, clf, classifier, data_type, subjects):
    # set the number of folds to 10
    n_folds = 10

    # get the groups for the cross-validation (in this case, the subjects)
    groups = subjects

    # create a GroupKFold cross-validator with the specified number of folds
    gkf = GroupKFold(n_splits=n_folds)
    gkf.get_n_splits(X, y, groups)

    # initialize lists to store the evaluation metrics for each fold
    confusion_matrix_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    # iterate over each fold
    for i in range(10):
        # iterate over each train-test split in the current fold
        for fold_index, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
            # if the current train-test split belongs to the current fold
            if i == fold_index:
                # split the data into training and testing sets based on the current train-test split
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # train the classifier on the training data and evaluate it on the testing data
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                cm, precision, recall, accuracy = PrintEvalMetrics(
                    [pred], [test_index], y, classifier, data_type, fold_index + 1)

                # append the evaluation metrics for the current fold to the respective lists
                confusion_matrix_scores.append(cm)
                precision_scores.append(precision)
                recall_scores.append(recall)
                accuracy_scores.append(accuracy)

                # update the training data to include the subjects that were not in the original training data
                groups_train = groups[train_index]
                groups_test = groups[test_index]
                new_groups = set(groups_test) - set(groups_train)
                new_index = [i for i, subj in enumerate(
                    groups) if subj in new_groups]
                X_train = np.concatenate((X_train, X[new_index]))
                y_train = np.concatenate((y_train, y[new_index]))

    # return the lists of evaluation metrics for all folds
    return confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores


def evaluate(X, y, classifier, data_type, subjects):
    clf = None
    if classifier == "SVM":
        clf = svm.LinearSVC(dual=False)
    elif classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "TREE":
        clf = DecisionTreeClassifier()

    confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = subject_independent_cross_validation(
        X, y, clf, classifier, data_type, subjects)

    # Compute and return the average score across all folds
    avg_cm = np.mean((confusion_matrix_scores), axis=0)
    avg_precision = np.mean((precision_scores), axis=0)
    avg_recall = np.mean((recall_scores), axis=0)
    avg_accuracy = np.mean((accuracy_scores), axis=0)

    # Save the average score across all folds in a file
    new_dir = 'Final_Outputs/'+classifier
    os.makedirs(new_dir, exist_ok=True)
    filename = classifier+"_"+data_type+".txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Average Scores \n')
            file.write("\tConfusion matrix: "+str(avg_cm) + '\n')
            file.write("\tPrecision: " + str(avg_precision) + '\n')
            file.write("\tRecall: " + str(avg_recall) + '\n')
            file.write("\tAccuracy: " + str(avg_accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)
