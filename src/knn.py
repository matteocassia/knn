import math


class KNN:

    def __init__(self, data, target, k):
        """Initialises a KNN classifier
        with training data as a dictionary containing class names and list of tuples containing numerical values
            (i.e. {"male_footballers": [(84.5, 181, 21, ...), ...], ...}),
        the target instance as a tuple containing numerical values
            (i.e. (75, 169, 34)) to be classified and
        the number of nearest neighbours to be considered as an integer
            (i.e. 3)."""
        self.data = data
        self.target = target
        self.k = k

    def fit(self):
        """Returns the string representing the name of the class
        which the target instance has been classified to."""
        neighbours = self.get_neighbours()
        nearest_neighbours = sorted(neighbours, key=lambda t: t[1])[: self.k]
        majority_class_name = self.get_majority_class_name(nearest_neighbours)
        return majority_class_name

    def get_neighbours(self):
        """It returns all the neighbours as tuples containing the class name and distance
        of the target instance."""
        neighbours = []
        for class_name in self.data:
            for instance in self.data[class_name]:
                neighbours.append((class_name, self.get_distance(self.target, instance)))
        return neighbours

    @staticmethod
    def get_majority_class_name(neighbours):
        """Given a list of neighbours as a tuple containing the class name and distance,
        it returns the most recurring class name of the neighbours."""
        class_names = [n[0] for n in neighbours]
        best_class_name = None
        occurrences = -1
        for class_name in class_names:
            x = class_names.count(class_name)
            if x > occurrences:
                best_class_name = class_name
                occurrences = x
        return best_class_name

    @staticmethod
    def get_distance(a, b):
        """Given two n-dimensional points as tuples of numerical values,
        it returns the (euclidean) distance between the two."""
        sum = 0
        for i in range(0, len(a)):
            d = pow(abs(a[i] - b[i]), 2)
            sum += d
        return math.sqrt(sum)
