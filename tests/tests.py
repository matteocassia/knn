import unittest
import knn as my_knn

class Test(unittest.TestCase):

    def setUp(self):
        samples = [
            ["bottom_left", (0.5, 1.0)],
            ["bottom_left", (1.5, 1.0)],
            ["bottom_left", (1.5, 1.5)],
            ["top_right", (9.5, 8.0)],
            ["top_right", (8.5, 7.0)],
            ["top_right", (10.5, 9.0)],
        ]
        k = 2
        target = (2.0, 3.0)
        knn = my_knn.KNN(samples, target, k)
        self.knn = knn

    def test_scale(self):
        self.assertEqual(self.knn.samples[0][1], (0.0, 0.0))
        self.assertEqual(self.knn.samples[1][1], (0.1, 0.0))
        self.assertEqual(self.knn.samples[2][1], (0.1, 0.0625))
        self.assertEqual(self.knn.samples[3][1], (0.9, 0.875))
        self.assertEqual(self.knn.samples[4][1], (0.8, 0.75))
        self.assertEqual(self.knn.samples[5][1], (1.0, 1.0))
        self.assertEqual(self.knn.target, (0.15, 0.25))

    def test_classify(self):
        self.assertEqual(self.knn.classify(), "bottom_left")

    def test_get_distance(self):
        point_a = (3.0, 4.0)
        point_b = (6.0, 0.0)
        self.assertEqual(self.knn.get_distance(point_a, point_b), 5.0)

    def test_get_majority_class(self):
        neighbours = [
            ["blue", (1.0, 2.0)],
            ["green", (1.0, 2.0)],
            ["green", (1.0, 2.0)],
            ["green", (1.0, 2.0)],
        ]
        self.assertEqual(self.knn.get_majority_class(neighbours), "green")