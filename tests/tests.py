import unittest
import knn as my_knn

class Test(unittest.TestCase):

    def setUp(self):
        data  = {
            "bottom_left": [
                (0.5, 1.0),
                (1.0, 1.5),
                (1.0, 2.0)],
            "top_right": [
                (9.5, 8.0),
                (8.0, 9.5),
                (9.0, 10.0)],
            "top_left": [
                (0.5, 8.0),
                (1.0, 9.5),
                (1.0, 10.0)]
        }
        k = 2
        target = (2.0, 9.0)
        knn = my_knn.KNN(data, target, k)
        self.knn = knn

    def test_get_majority_class(self):
        self.assertEqual(self.knn.fit(), "top_left")

    def test_get_distance(self):
        point_a = (3.0, 4.0)
        point_b = (6.0, 0.0)
        self.assertEqual(self.knn.get_distance(point_a, point_b), 5.0)

    def test_get_majority_class_name(self):
        neighbours = [
            ("blue", (1.0, 2.0)),
            ("green", (1.0, 2.0)),
            ("green", (1.0, 2.0)),
            ("green", (1.0, 2.0)),
        ]
        self.assertEqual(self.knn.get_majority_class_name(neighbours), "green")
