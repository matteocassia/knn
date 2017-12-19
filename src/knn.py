import math


class KNN:

    def __init__(self, samples, target, k):
        """Initialises a KNN classifier
        with training data as a list containing lists containing the classes and a tuple containing numerical values
            (i.e. [["male_footballer": (84.5, 181, 21)], ...]),
        the target instance as a tuple containing numerical values
            (i.e. (75, 169, 34)) to be classified and
        the number of nearest neighbours to be considered as an integer
            (i.e. 3)."""
        self.samples = samples
        self.target = target
        self.k = k
        self.scale()

    def scale(self):
        self.samples.insert(0, ["target", self.target])
        for i in range(0, len(self.samples[1])):
            min = float("+Inf")
            max = float("-Inf")
            for sample in self.samples:
                if sample[1][i] < min:
                    min = sample[1][i]
                elif sample[1][i] > max:
                    max = sample[1][i]
            for sample in self.samples:
                a = list(sample[1])
                a[i] = (a[i] - min) / (max - min)
                sample[1] = tuple(a)
        self.target = self.samples[0][1]
        self.samples = self.samples[1:]

    def classify(self):
        """Returns the class which the target instance has been classified to."""
        sample_distances = self.get_sample_distances()
        nearest_neighbours = map(lambda a: a[1], sample_distances[:self.k])
        majority_class = self.get_majority_class(nearest_neighbours)
        return majority_class

    def get_sample_distances(self):
        """It returns all the samples associated with their distance to the target
        as a list containing distance and sample."""
        sample_distances = []
        for sample in self.samples:
            distance = self.get_distance(self.target, sample[1])
            sample_distances.append((distance, sample))
        return sorted(sample_distances)

    @staticmethod
    def get_majority_class(neighbours):
        """Given a list of samples,
        it returns the most recurring class name of the neighbours."""
        counts = {}
        for neighbour in neighbours:
            if neighbour[0] not in counts:
                counts[neighbour[0]] = 0
            counts[neighbour[0]] += 1
        majority_count = -1
        majority_class = None
        for c in counts:
            if counts[c] > majority_count:
                majority_count = counts[c]
                majority_class = c
        return majority_class

    @staticmethod
    def get_distance(a, b):
        """Given two n-dimensional points as tuples of numerical values,
        it returns the (euclidean) distance between the two."""
        t = 0
        for i in range(0, len(a)):
            d = pow(abs(a[i] - b[i]), 2)
            t += d
        return math.sqrt(t)