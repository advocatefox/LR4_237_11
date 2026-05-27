import numpy as np

def gini(y):
    y = np.array(y)
    if len(y) == 0:
        return 0

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)

    if len(np.unique(feature_vector)) <= 1:
        return None, None, None, None

    sorted_idx = np.argsort(feature_vector)
    x = feature_vector[sorted_idx]
    y = target_vector[sorted_idx]

    thresholds = (x[:-1] + x[1:]) / 2

    best_threshold = None
    best_gini = np.inf
    best_idx = None

    ginis = []

    n = len(y)

    for i, t in enumerate(thresholds):

        left_mask = x < t
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            ginis.append(np.inf)
            continue

        y_left = y[left_mask]
        y_right = y[right_mask]

        curr_gini = (
            len(y_left) / n * gini(y_left)
            + len(y_right) / n * gini(y_right)
        )

        ginis.append(curr_gini)

        if curr_gini < best_gini:
            best_gini = curr_gini
            best_threshold = t
            best_idx = i

    return thresholds, np.array(ginis), best_threshold, best_gini

class DecisionTree:

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._fit_node(X, y, depth=0)

    def _fit_node(self, X, y, depth):

        # stop 1: pure node
        if len(np.unique(y)) == 1:
            return {
                'type': 'terminal',
                'class': y[0],
                'y': y
            }

        if self.max_depth is not None and depth >= self.max_depth:
            return {
                'type': 'terminal',
                'class': self._majority_class(y),
                'y': y
            }

        n_features = X.shape[1]

        best_feature = None
        best_threshold = None
        best_gini = np.inf
        best_split = None

        for feature in range(n_features):

            thresholds, ginis, thr, g = find_best_split(
                X[:, feature],
                y
            )

            if thr is None:
                continue

            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = thr

        if best_feature is None:
            return {
                'type': 'terminal',
                'class': self._majority_class(y),
                'y': y
            }

        feature_values = X[:, best_feature]

        if self._is_numeric(feature_values):
            left_mask = feature_values < best_threshold
        else:
            left_mask = feature_values == best_threshold

        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {
                'type': 'terminal',
                'class': self._majority_class(y),
                'y': y
            }

        node = {
            'type': 'node',
            'feature_split': best_feature,
            'threshold': best_threshold,
            'feature_type': 'real',
            'left_child': self._fit_node(X[left_mask], y[left_mask], depth + 1),
            'right_child': self._fit_node(X[right_mask], y[right_mask], depth + 1)
        }

        return node

    def predict(self, X):
        return np.array([self._predict_node(x, self.tree) for x in X])

    def _predict_node(self, x, node):

        if node['type'] == 'terminal':
            # majority class in leaf
            values, counts = np.unique(node['y'], return_counts=True)
            return values[np.argmax(counts)]

        feature = node['feature_split']

        if node['feature_type'] == 'real':
            if x[feature] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

        else:
            if x[feature] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _is_numeric(self, x):
        return np.issubdtype(np.array(x).dtype, np.number)