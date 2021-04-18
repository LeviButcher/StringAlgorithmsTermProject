from utils import measure_distance_performance, generate_confusion_matrix
import numpy as np


class TestUtils:
    def test_generate_confusion_matrix_simple_case(self):
        testDataset = [([], [([], True), ([], False)])]
        expected_cm = [[1, 0], [0, 1]]
        predicted = [[4], []]

        res_cm = generate_confusion_matrix(testDataset, predicted)

        assert np.array_equal(res_cm, expected_cm)

    def test_generate_confusion_matrix_complex_case(self):
        testDataset = [([], [([], True), ([], False), ([], True)]),
                       ([], [([], False), ([], False), ([], True)])]

        # all correct except one True Negative
        predicted = [[2, 4], [], [5], [1], [], [2]]

        expected_cm = [[2, 1], [0, 3]]

        res_cm = generate_confusion_matrix(testDataset, predicted)

        assert np.array_equal(res_cm, expected_cm)

    def test_measure_distance_performance(self, mocker):
        # Call measure_distance with fake datset
        # mock findVideoSeq to return certain results
        # computer confusion matrix based on results within function
        # then check to see if resulting confusion matrix is expected
        l = 0.1
        d = 0.1

        def fakeDistance(f, ff): return .5

        testDataset = [([], [([], False), ([], True), ([], False)]),
                       ([], [([], False), ([], False), ([], False)])]

        expected_cm = [[0, 5], [0, 1]]

        mocker.patch('video_matching.findVideoSeq', return_value=[1, 2])

        res_cm = measure_distance_performance(l, d, fakeDistance, testDataset)

        assert np.array_equal(res_cm, expected_cm)
