from hmmlearn import hmm


class HMMClassifier:
    def __init__(self):
        self._hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

    def train(self, train_x):
        self._hmm_model.fit(train_x)

    def test(self, test_x):
        return self._hmm_model.predict(test_x)

    def accuracy(self, x_test, y_test):
        y_pred = self._hmm_model.predict(x_test)
        print(y_pred)
        print(y_test)
        return HMMClassifier._accuracy(y_pred, y_test)

    @staticmethod
    def _accuracy(y_prediction, y_test):
        assert (len(y_prediction) == len(y_test))
        test_data_count = len(y_test)
        correct_classification_count = len([i for i, j in zip(y_prediction, y_test) if i == int(j)])
        return correct_classification_count / test_data_count