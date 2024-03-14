import sys
import numpy as np
from classification import evaluate
from data_processing import read_data
import random
import matplotlib.pyplot as plot


def run_experiment(classifier, data_type, data_directory):

    print("------------------------------------------------------")
    print(classifier, data_type)

    # Read and transform the dataset
    X, y, subjects = read_data(data_directory, data_type)

    X = np.array(X)
    y = np.array(y)
    subjects = np.array(subjects)

    if data_type == "Original" or "Translated":
        evaluate(X, y, classifier, data_type, subjects)

    elif data_type == "Rotated":
        X_rotated_x, X_rotated_y, X_rotated_z = X[:, 0], X[:, 1], X[:, 2]
        # for RotatedX
        evaluate(X_rotated_x, y, classifier, data_type+"X", subjects)
        # for RotatedY
        evaluate(X_rotated_y, y, classifier, data_type+"Y", subjects)
        # for RotatedZ
        evaluate(X_rotated_z, y, classifier, data_type+"Z", subjects)

    print("------------------------------------------------------")


def plot_sample_points(X, label, color):
    # Take a random sample from datatset to plot the face
    sample_index = random.randint(0, len(X))
    X = X[sample_index].reshape(83, 3)

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, label=label)

    # Add labels for each axis
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Add a legend to the plot
    ax.legend()

    # Display the plot
    plot.show()


if __name__ == "__main__":
    # Command Line Arguments
    classifier = sys.argv[1]
    data_type = sys.argv[2]
    data_directory = sys.argv[3]

    # Check if classifier argument is valid
    if classifier not in ["SVM", "RF", "TREE"]:
        print("Invalid classifier")
        sys.exit()

    # Check if data type argument is valid
    if data_type not in ["Original", "Translated", "Rotated"]:
        print("Invalid Datatype")
        sys.exit()

    # Run experiment
    run_experiment(classifier, data_type, data_directory)
