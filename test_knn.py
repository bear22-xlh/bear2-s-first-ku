"""knn.py 的单元测试：距离函数与最近邻搜索。"""

from __future__ import annotations

import unittest

import numpy as np

from knn import (
    KNNClassifier,
    KNNRegressor,
    accuracy_score,
    classification_report,
    confusion_matrix,
    cross_validate_knn,
    euclidean_distance,
    majority_vote,
    manhattan_distance,
    nearest_neighbors,
    pairwise_distances,
    pairwise_distances_loop,
    stratified_kfold_indices,
    stratified_split_indices,
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


class TestStratifiedSplitIndices(unittest.TestCase):
    def test_split_indices_reproducible(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        train_1, test_1 = stratified_split_indices(y, test_size=0.2, random_state=42)
        train_2, test_2 = stratified_split_indices(y, test_size=0.2, random_state=42)

        np.testing.assert_array_equal(train_1, train_2)
        np.testing.assert_array_equal(test_1, test_2)

    def test_split_indices_keep_class_ratio(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        train_index, test_index = stratified_split_indices(
            y, test_size=0.2, random_state=42
        )

        self.assertEqual(len(train_index), 16)
        self.assertEqual(len(test_index), 4)
        self.assertEqual(int(np.sum(y[test_index] == "a")), 2)
        self.assertEqual(int(np.sum(y[test_index] == "b")), 2)


class TestKNNRegressor(unittest.TestCase):
    def test_uniform_mean_prediction(self) -> None:
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([10.0, 20.0, 30.0])
        regressor = KNNRegressor(k=2, weights="uniform").fit(X_train, y_train)

        prediction = regressor.predict(np.array([[0.9]]))[0]
        self.assertAlmostEqual(prediction, 15.0)

    def test_distance_weighted_prediction(self) -> None:
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([10.0, 20.0, 30.0])
        regressor = KNNRegressor(k=2, weights="distance").fit(X_train, y_train)

        prediction = regressor.predict(np.array([[0.9]]))[0]
        self.assertAlmostEqual(prediction, 19.0, places=6)

    def test_exact_match_dominates_distance_weighting(self) -> None:
        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([10.0, 20.0])
        regressor = KNNRegressor(k=2, weights="distance").fit(X_train, y_train)

        prediction = regressor.predict(np.array([[0.0]]))[0]
        self.assertAlmostEqual(prediction, 10.0, places=6)

    def test_predict_before_fit_raises(self) -> None:
        regressor = KNNRegressor(k=1)
        with self.assertRaises(RuntimeError):
            regressor.predict(np.array([[0.0]]))

    def test_k_too_large_raises(self) -> None:
        regressor = KNNRegressor(k=3)
        with self.assertRaises(ValueError):
            regressor.fit(
                np.array([[0.0], [1.0]]),
                np.array([10.0, 20.0]),
            )

    def test_invalid_weights_raises(self) -> None:
        with self.assertRaises(ValueError):
            KNNRegressor(k=1, weights="linear")


class TestEvaluationMetrics(unittest.TestCase):
    def test_accuracy_score(self) -> None:
        y_true = np.array(["a", "b", "c"])
        y_pred = np.array(["a", "b", "a"])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 2 / 3)

    def test_confusion_matrix(self) -> None:
        y_true = np.array(["a", "a", "b", "b"])
        y_pred = np.array(["a", "b", "a", "b"])
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred, labels=["a", "b"]), expected
        )

    def test_confusion_matrix_label_order(self) -> None:
        y_true = np.array(["a", "a", "a", "b"])
        y_pred = np.array(["a", "a", "b", "b"])
        expected = np.array([[1, 0], [1, 2]])
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred, labels=["b", "a"]), expected
        )

    def test_classification_report_contains_metrics(self) -> None:
        report = classification_report(
            np.array(["a", "a", "b", "b"]),
            np.array(["a", "b", "a", "b"]),
            labels=["a", "b"],
        )
        self.assertIn("precision", report)
        self.assertIn("recall", report)
        self.assertIn("f1", report)
        self.assertIn("accuracy: 0.5000", report)

    def test_classification_report_handles_unpredicted_class(self) -> None:
        report = classification_report(
            np.array(["a", "a", "b"]),
            np.array(["a", "a", "a"]),
            labels=["a", "b"],
        )
        self.assertIn("b", report)
        self.assertIn("0.0000", report)

    def test_accuracy_score_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            accuracy_score(np.array(["a", "b"]), np.array(["a"]))

    def test_confusion_matrix_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            confusion_matrix(np.array(["a", "b"]), np.array(["a"]))


class TestStratifiedKFold(unittest.TestCase):
    def test_each_sample_in_validation_once(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        splits = stratified_kfold_indices(y, n_splits=5, random_state=42)
        counts = np.zeros(len(y), dtype=int)
        for _, validation_index in splits:
            counts[validation_index] += 1
        np.testing.assert_array_equal(counts, np.ones(len(y), dtype=int))

    def test_fold_sizes_and_class_balance(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        splits = stratified_kfold_indices(y, n_splits=5, random_state=42)
        for train_index, validation_index in splits:
            self.assertEqual(len(train_index), 16)
            self.assertEqual(len(validation_index), 4)
            self.assertEqual(int(np.sum(y[validation_index] == "a")), 2)
            self.assertEqual(int(np.sum(y[validation_index] == "b")), 2)

    def test_folds_reproducible(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        splits_1 = stratified_kfold_indices(y, n_splits=5, random_state=42)
        splits_2 = stratified_kfold_indices(y, n_splits=5, random_state=42)
        for (train_1, validation_1), (train_2, validation_2) in zip(
            splits_1, splits_2
        ):
            np.testing.assert_array_equal(train_1, train_2)
            np.testing.assert_array_equal(validation_1, validation_2)

    def test_too_many_splits_raises(self) -> None:
        y = np.array(["a"] * 10 + ["b"] * 10)
        with self.assertRaises(ValueError):
            stratified_kfold_indices(y, n_splits=11, random_state=42)


class TestCrossValidateKNN(unittest.TestCase):
    def test_returns_all_k_values(self) -> None:
        X = np.array([[0.0], [0.1], [0.2], [0.3], [10.0], [10.1], [10.2], [10.3]])
        y = np.array(["a"] * 4 + ["b"] * 4)
        results = cross_validate_knn(
            X, y, k_values=[1, 3], n_splits=2, random_state=42
        )

        self.assertEqual(set(results), {1, 3})
        for k in (1, 3):
            self.assertEqual(len(results[k]["fold_accuracies"]), 2)
            self.assertGreaterEqual(results[k]["mean_accuracy"], 0.0)
            self.assertLessEqual(results[k]["mean_accuracy"], 1.0)

    def test_k1_separates_simple_data(self) -> None:
        X = np.array([[0.0], [0.1], [0.2], [0.3], [10.0], [10.1], [10.2], [10.3]])
        y = np.array(["a"] * 4 + ["b"] * 4)
        results = cross_validate_knn(
            X, y, k_values=[1], n_splits=2, random_state=42
        )
        self.assertAlmostEqual(results[1]["mean_accuracy"], 1.0)

    def test_invalid_k_raises(self) -> None:
        X = np.array([[0.0], [1.0]])
        y = np.array(["a", "b"])
        with self.assertRaises(ValueError):
            cross_validate_knn(X, y, k_values=[0], n_splits=2, random_state=42)


class TestPairwiseDistancesLoop(unittest.TestCase):
    def test_loop_matches_vectorized_euclidean(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 3))
        Y = rng.normal(size=(4, 3))
        np.testing.assert_allclose(
            pairwise_distances_loop(X, Y),
            pairwise_distances(X, Y),
        )

    def test_loop_matches_vectorized_manhattan(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5, 2))
        Y = rng.normal(size=(3, 2))
        np.testing.assert_allclose(
            pairwise_distances_loop(X, Y, metric="manhattan"),
            pairwise_distances(X, Y, metric="manhattan"),
        )

    def test_loop_rejects_unknown_metric(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_distances_loop(
                np.zeros((2, 2)), np.zeros((2, 2)), metric="cosine"
            )

    def test_loop_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_distances_loop(np.zeros((2, 3)), np.zeros((2, 4)))


class TestAdditionalValidation(unittest.TestCase):
    def test_classifier_rejects_empty_training_data(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=1).fit(np.empty((0, 4)), np.array([]))

    def test_classifier_rejects_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)

    def test_classifier_rejects_invalid_metric(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=1, metric="cosine")


if __name__ == "__main__":
    unittest.main()
