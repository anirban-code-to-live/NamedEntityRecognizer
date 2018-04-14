from collections import Counter
# import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import CRF
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


class CRFClassifier:
    def __init__(self):
        self._crf_model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self._params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        self._best_crf_model = None
        self._labels = None

    def train(self, x_train,y_train):
        self._crf_model.fit(x_train, y_train)
        self._labels = list(self._crf_model.classes_)
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=self._labels)
        self._best_crf_model = RandomizedSearchCV(self._crf_model, self._params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=2,
                                scoring=f1_scorer)
        self._best_crf_model.fit(x_train, y_train)

        print('best params:', self._best_crf_model.best_params_)
        print('best CV score:', self._best_crf_model.best_score_)
        print('model size: {:0.2f}M'.format(self._best_crf_model.best_estimator_.size_ / 1000000))

    def test(self, test_x):
        return self._crf_model.predict(test_x)

    def get_average_f1_score(self, x_test, y_test):
        # y_pred = self._crf_model.predict(x_test)
        crf = self._best_crf_model.best_estimator_
        y_pred = crf.predict(x_test)
        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=self._labels)

    def accuracy(self, x_test, y_test):
        y_pred = self._crf_model.predict(x_test)
        return CRFClassifier.__accuracy(y_pred, y_test)

    def get_class_f1_score(self, x_test, y_test):
        # y_pred = self._crf_model.predict(x_test)
        crf = self._best_crf_model.best_estimator_
        y_pred = crf.predict(x_test)
        labels = list(self._crf_model.classes_)
        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        return metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3)

    # def visualize_parameter_space(self):
    #     plt.style.use('ggplot')
    #     _x = [s.parameters['c1'] for s in self._best_crf_model.grid_scores_]
    #     _y = [s.parameters['c2'] for s in self._best_crf_model.grid_scores_]
    #     _c = [s.mean_validation_score for s in self._best_crf_model.grid_scores_]
    #
    #     fig = plt.figure()
    #     fig.set_size_inches(12, 12)
    #     ax = plt.gca()
    #     ax.set_yscale('log')
    #     ax.set_xscale('log')
    #     ax.set_xlabel('C1')
    #     ax.set_ylabel('C2')
    #     ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    #         min(_c), max(_c)
    #     ))
    #     ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])
    #     print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    def print_top_positive_features(self):
        print("Top positive:")
        CRFClassifier.__print_state_features(Counter(self._best_crf_model.state_features_).most_common(30))

    def print_top_negative_features(self):
        print("\nTop negative:")
        CRFClassifier.__print_state_features(Counter(self._best_crf_model.state_features_).most_common()[-30:])

    @staticmethod
    def __print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    @staticmethod
    def __accuracy(y_prediction, y_test):
        assert (len(y_prediction) == len(y_test))
        test_data_count = len(y_test)
        correct_classification_count = len([i for i, j in zip(y_prediction, y_test) if i == int(j)])
        return correct_classification_count / test_data_count