import knn
import math

# Opens the file containing the data and extracts sex, height and weight;
# The resulting list is in the form [["sex", (height, weight)], ...]

a = []
with open("../data/athletes.csv", "r") as file:
    lines = file.readlines()
    for line in lines:
        elements = line.split(",")
        try:
            sex = str(elements[3])
            height = float(elements[5])
            weight = float(elements[6])
            a.append([sex, (height, weight)])
        except(ValueError, TypeError):
            print("")

# Tests and cross-validates the accuracy of the classifier
t = 0
f = 0
for athlete in a:
    target = athlete
    samples = list(a)
    samples.remove(target)
    k = 10
    kn = knn.KNN(samples, target[1], k)
    c = kn.classify()
    if c == target[0]:
        t += 1
    else:
        f += 1
    acc = math.ceil((t / (t + f)) * 100)
    p = math.ceil(100 * (t + f) / len(a))
    print("At " + str(p) + "%, accuracy is " + str(acc) + "%")
