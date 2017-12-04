import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


def load_data(file_name):
    data = []
    with open(file_name, 'rU') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)

    return data


def convert_str_to_float(data):
    converted_data = []
    for row in data:
        converted_row = [float(item.strip()) for item in row]
        converted_data.append(converted_row)

    return converted_data


def get_xy(data):
    x, y = [], []
    for row in data:
        x.append(row[0:-1])
        y.append(row[-1])

    return x, y


def convert_to_label(y):
    converted_y = pd.qcut(y, 5, labels=False)

    return converted_y


def insert_probabilities(x, probabilities):
    new_x = []
    for row, prob in zip(x, probabilities):
        new_x.append(np.concatenate([row, prob]))

    return new_x


def naive_bayes(train_x, test_x, train_y, test_y):
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)
    prediction = gnb.predict(test_x)

    return prediction


def decision_tree(train_x, test_x, train_y, test_y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)

    return prediction


def hybrid(train_x, test_x, train_y, test_y):
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)

    probabilities = gnb.predict_proba(train_x)
    train_x = insert_probabilities(train_x, probabilities)

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)

    probabilities = gnb.predict_proba(test_x)
    test_x = insert_probabilities(test_x, probabilities)

    prediction = clf.predict(test_x)

    return prediction


def get_accuracy(test_y, prediction):
    correct = 0
    for actual, predicted in zip(test_y, prediction):
        if actual == predicted:
            correct += 1

    accuracy = str(correct / len(test_y) * 100)

    return accuracy


def main():
    data = load_data("ALL.csv")
    data = convert_str_to_float(data)
    x, y = get_xy(data)
    y = convert_to_label(y)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.30, random_state=42)

    hybrid_prediction = hybrid(train_x, test_x, train_y, test_y)
    hybrid_accuracy = get_accuracy(test_y, hybrid_prediction)

    dt_prediction = decision_tree(train_x, test_x, train_y, test_y)
    dt_accuracy = get_accuracy(test_y, dt_prediction)

    nb_prediction = naive_bayes(train_x, test_x, train_y, test_y)
    nb_accuracy = get_accuracy(test_y, nb_prediction)

    print("HYBRID:", hybrid_accuracy)
    print("DECISION TREE:", dt_accuracy)
    print("NAIVE BAYES:", nb_accuracy)


main()
