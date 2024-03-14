import zipfile
import os
import numpy as np
from math import acos


def get_class_label(label):
    class_labels = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5
    }
    return class_labels.get(label)


def translate_landmarks(landmarks):
    # Convert the list to a NumPy array
    landmarks = np.array(landmarks)

    # Calculate the mean of each x, y, z
    mean_x = np.mean(landmarks[:, 0])
    mean_y = np.mean(landmarks[:, 1])
    mean_z = np.mean(landmarks[:, 2])

    # Subtract the mean from each x, y, z coordinate
    landmarks[:, 0] -= mean_x
    landmarks[:, 1] -= mean_y
    landmarks[:, 2] -= mean_z

    return landmarks.flatten()


def rotate_landmarks(landmarks):
    landmarks = np.array(landmarks)
    # Calculate the value of pi
    pi = round(2 * acos(0.0), 3)

    cos = np.cos(pi)
    sine = np.sin(pi)

    # Respective Axis data for rotating the coordinates
    x_axis = np.array([[1, 0, 0], [0, cos, sine], [0, -sine, cos]])
    y_axis = np.array([[cos, 0, -sine], [0, 1, 0], [sine, 0, cos]])
    z_axis = np.array([[cos, sine, 0], [-sine, cos, 0], [0, 0, 1]])

    # Rotate the points around each axis
    rotated_x = x_axis.dot(landmarks.T).T
    rotated_y = y_axis.dot(landmarks.T).T
    rotated_z = z_axis.dot(landmarks.T).T

    return rotated_x.flatten(), rotated_y.flatten(), rotated_z.flatten()


def read_data(path, data_type):
    # Initialize empty lists to store features, classes, and subjects
    features = []
    classes = []
    subjects = []

    # Check if path is a zip file
    if path.endswith('.zip'):
        # create a ZipFile object and extract all files to memory
        with zipfile.ZipFile(path) as myzip:
            # Loop over all subject directories, expressions, and bnd files in the zip file
            for subject_dir in sorted(myzip.namelist()):
                for expression_dir in sorted(myzip.namelist()):
                    for bnd_file in sorted(myzip.namelist()):
                        # Check if file has .bnd extension
                        if bnd_file.endswith('.bnd'):
                            # Extract the class label i.e Angry, Sad, Happy etc
                            label = os.path.basename(os.path.dirname(bnd_file))
                            label_val = get_class_label(label)
                            # Extract the subject
                            subject = os.path.basename(
                                os.path.dirname(os.path.dirname(bnd_file)))
                            # Extract the file to memory
                            bnd_data = myzip.read(bnd_file)
                            # Convert the bytes to a string
                            bnd_str = bnd_data.decode('utf-8')
                            landmarks = []
                            # Extract the coordinates in string i.e x,y,z
                            for line in bnd_str.split("\n"):
                                if len(line) > 0:
                                    x, y, z = line.split()[1:]
                                    landmarks.append(
                                        [float(x), float(y), float(z)])

                            if data_type == "Original":
                                landmarks = np.array(landmarks).flatten()
                            elif data_type == "Translated":
                                landmarks = translate_landmarks(landmarks)
                            elif data_type == "Rotated":
                                landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotate_landmarks(
                                    landmarks)
                                landmarks = [landmarksRotatedX,
                                             landmarksRotatedY, landmarksRotatedZ]

                            # Append the features, class labels, and subject to their respective lists
                            features.append(landmarks)
                            classes.append(label_val)
                            subjects.append(subject)
    else:
        # Loop over all subject directories and bnd files in the given path
        for subdir, dirs, files in os.walk(path):
            for file in files:
                # Check if file has .bnd extension
                if file.endswith('.bnd'):
                    # Extract the class label i.e Angry, Sad, Happy etc
                    label = subdir.split(os.path.sep)[-1]
                    label_val = get_class_label(label)
                    # Extract the subject
                    subject = os.path.join(subdir, file)
                    # File path
                    filepath = os.path.join(subdir, file)
                    with open(filepath, 'r') as f:
                        landmarks = []
                        # Extract the lines from the file
                        bnd_data = f.readlines()
                        # Extract the coordinates in string i.e x,y,z
                        for line in bnd_data[:84]:
                            x, y, z = line.split()[1:]
                            landmarks.append([float(x), float(y), float(z)])

                        if data_type == "Original":
                            landmarks = np.array(landmarks).flatten()
                        elif data_type == "Translated":
                            landmarks = translate_landmarks(landmarks)
                        elif data_type == "Rotated":
                            landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotate_landmarks(
                                landmarks)
                            landmarks = [landmarksRotatedX,
                                         landmarksRotatedY, landmarksRotatedZ]

                            # Append the features, class labels, and subject to their respective lists
                        features.append(landmarks)
                        classes.append(label_val)
                        subjects.append(subject)
    return features, classes, subjects
