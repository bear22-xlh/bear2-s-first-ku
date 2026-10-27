"""knn.py 的单元测试：距离函数与最近邻搜索。"""

from __future__ import annotations

import unittest

import numpy as np

from knn import (
    KNNClassifier,
    euclidean_distance,
    majority_vote,
    manhattan_distance,
    nearest_neighbors,
    pairwise_distances,
)


class TestDistanceFunctions(unittest.TestCase):
    def test_euclidean_distance_hand_calculation(self) -> None:
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(euclidean_distance(x1, x2), 5.0)

    def test_manhattan_distance_hand_calculation(self) -> None:
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(manhattan_distance(x1, x2), 7.0)

    def test_pairwise_euclidean_matches_hand_calculation(self) -> None:
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        Y = np.array([[0.0, 0.0], [3.0, 4.0]])
        expected = np.array([[0.0, 5.0], [5.0, 0.0]])
        np.testing.assert_allclose(pairwise_distances(X, Y), expected)

    def test_pairwise_manhattan_matches_hand_calculation(self) -> None:
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        Y = np.array([[0.0, 0.0], [3.0, 4.0]])
        expected = np.array([[0.0, 7.0], [7.0, 0.0]])
        np.testing.assert_allclose(
            pairwise_distances(X, Y, metric="manhattan"), expected
        )

    def test_pairwise_euclidean_matches_numpy_formula(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 3))
        Y = rng.normal(size=(4, 3))
        expected = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        np.testing.assert_allclose(pairwise_distances(X, Y), expected)

    def test_pairwise_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_distances(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_pairwise_rejects_unknown_metric(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_distances(np.zeros((2, 3)), np.zeros((2, 3)), metric="cosine")


class TestNearestNeighbors(unittest.TestCase):
    def test_nearest_neighbors_sorted_by_distance(self) -> None:
        X_train = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        query = np.array([0.0, 0.0])
        indices, distances = nearest_neighbors(X_train, query, k=2)
        np.testing.assert_array_equal(indices, np.array([[0, 1]]))
        np.testing.assert_allclose(distances, np.array([[1.0, 2.0]]))

    def test_nearest_neighbors_manhattan(self) -> None:
        X_train = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        query = np.array([0.0, 0.0])
        indices, distances = nearest_neighbors(X_train, query, k=2, metric="manhattan")
        np.testing.assert_array_equal(indices, np.array([[0, 1]]))
        np.testing.assert_allclose(distances, np.array([[2.0, 2.0]]))

    def test_nearest_neighbors_multiple_queries(self) -> None:
        X_train = np.array([[0.0, 0.0], [10.0, 0.0]])
        X_query = np.array([[0.5, 0.0], [9.0, 0.0]])
        indices, distances = nearest_neighbors(X_train, X_query, k=1)
        np.testing.assert_array_equal(indices, np.array([[0], [1]]))
        np.testing.assert_allclose(distances, np.array([[0.5], [1.0]]))

    def test_nearest_neighbors_rejects_too_large_k(self) -> None:
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        query = np.array([0.0, 0.0])
        with self.assertRaises(ValueError):
            nearest_neighbors(X_train, query, k=3)

    def test_nearest_neighbors_rejects_dimension_mismatch(self) -> None:
        X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
        query = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            nearest_neighbors(X_train, query, k=1)


class TestMajorityVote(unittest.TestCase):
    def test_clear_majority(self) -> None:
        labels = np.array(["a", "a", "b"])
        distances = np.array([0.1, 0.2, 0.3])
        self.assertEqual(majority_vote(labels, distances), "a")

    def test_tie_broken_by_nearest_distance(self) -> None:
        labels = np.array(["a", "b"])
        distances = np.array([0.4, 0.1])
        self.assertEqual(majority_vote(labels, distances), "b")

    def test_three_way_tie_uses_nearest(self) -> None:
        labels = np.array(["a", "b", "c"])
        distances = np.array([0.5, 0.1, 0.2])
        self.assertEqual(majority_vote(labels, distances), "b")

    def test_empty_labels_raises(self) -> None:
        with self.assertRaises(ValueError):
            majority_vote(np.array([]))

    def test_distance_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            majority_vote(np.array(["a", "b"]), np.array([0.1]))

    def test_tie_without_distances_raises(self) -> None:
        with self.assertRaises(ValueError):
            majority_vote(np.array(["a", "b"]))


class TestKNNClassifier(unittest.TestCase):
    def test_predicts_simple_two_class(self) -> None:
        X_train = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0], [10.2, 10.0]])
        y_train = np.array(["a", "a", "b", "b"])
        classifier = KNNClassifier(k=1).fit(X_train, y_train)

        predictions = classifier.predict(
            np.array([[0.05, 0.0], [10.05, 10.0]])
        )
        np.testing.assert_array_equal(predictions, np.array(["a", "b"]))

    def test_k3_majority_vote(self) -> None:
        X_train = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]])
        y_train = np.array(["a", "a", "b"])
        classifier = KNNClassifier(k=3).fit(X_train, y_train)

        self.assertEqual(classifier.predict(np.array([[0.2, 0.0]]))[0], "a")

    def test_predict_before_fit_raises(self) -> None:
        classifier = KNNClassifier(k=1)
        with self.assertRaises(RuntimeError):
            classifier.predict(np.array([[0.0, 0.0]]))

    def test_k_too_large_raises(self) -> None:
        classifier = KNNClassifier(k=3)
        with self.assertRaises(ValueError):
            classifier.fit(
                np.array([[0.0, 0.0], [1.0, 1.0]]),
                np.array(["a", "b"]),
            )

    def test_predict_dimension_mismatch_raises(self) -> None:
        classifier = KNNClassifier(k=1).fit(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array(["a", "b"]),
        )
        with self.assertRaises(ValueError):
            classifier.predict(np.array([[0.0, 0.0, 0.0]]))

    def test_predict_empty_raises(self) -> None:
        classifier = KNNClassifier(k=1).fit(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array(["a", "b"]),
        )
        with self.assertRaises(ValueError):
            classifier.predict(np.empty((0, 2)))


if __name__ == "__main__":
    unittest.main()
