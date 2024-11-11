import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class Relation(Enum):
    EXISTS = 'exists'
    FORALL = 'forall'
    EVENTUALLY = 'eventually'
    ALWAYS = 'always'
    UNTIL = 'until'
    NEXT = 'next'
    PREVIOUS = 'previous'
    SINCE = 'since'
    BEFORE = 'before'
    AFTER = 'after'

@dataclass
class Decision:
    feature: int
    relation: Relation
    threshold: float
    secondary_feature: int = None
    secondary_threshold: float = None
    time_window: int = None

    def __str__(self):
        base_str = f"{self.relation.value}(feature{self.feature} >= {self.threshold:.2f}"
        if self.relation in [Relation.UNTIL, Relation.SINCE]:
            return f"{base_str} {self.relation.value} feature{self.secondary_feature} >= {self.secondary_threshold:.2f})"
        elif self.relation in [Relation.NEXT, Relation.PREVIOUS]:
            return f"{base_str}, time_window={self.time_window})"
        elif self.relation in [Relation.BEFORE, Relation.AFTER]:
            return f"{base_str} {self.relation.value} time={self.time_window})"
        else:
            return f"{base_str})"

class DecisionNode:
    def __init__(self, decision: Decision, left, right):
        self.decision = decision
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self, value):
        self.value = value

class ModalDecisionTreeBase(BaseEstimator):
    def __init__(self, max_depth=5, n_subfeatures=1, n_subrelations=2, min_samples_leaf=10):
        self.max_depth = max_depth
        self.n_subfeatures = n_subfeatures
        self.n_subrelations = n_subrelations
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, allow_nd=True)
        if X.ndim != 3:
            raise ValueError("X should be a 3D array: (n_samples, n_features, n_timepoints)")
        
        self.n_features_in_ = X.shape[1]
        self.n_timepoints_ = X.shape[2]
        
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            return LeafNode(self._leaf_value(y))
        
        best_score = float('inf')
        best_decision = None
        
        for decision in self._generate_relevant_decisions(X):
            score = self._evaluate_split(X, y, decision)
            if score < best_score:
                best_score = score
                best_decision = decision
        
        if best_decision is None:
            return LeafNode(self._leaf_value(y))
        
        mask = self._apply_decision(X, best_decision)
        
        left = self._build_tree(X[mask], y[mask], depth+1)
        right = self._build_tree(X[~mask], y[~mask], depth+1)
        
        return DecisionNode(best_decision, left, right)

    def _generate_relevant_decisions(self, X):
        n_samples, n_features, n_timepoints = X.shape
        features = np.random.choice(n_features, self.n_subfeatures, replace=False)
        relations = np.random.choice(list(Relation), self.n_subrelations, replace=False)
        decisions = []
        
        for feature in features:
            for relation in relations:
                unique_values = np.unique(X[:, feature, :])
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                for threshold in thresholds:
                    if relation in [Relation.UNTIL, Relation.SINCE]:
                        secondary_feature = np.random.choice(n_features)
                        secondary_threshold = np.random.choice(thresholds)
                        decisions.append(Decision(feature, relation, threshold, secondary_feature, secondary_threshold))
                    elif relation in [Relation.NEXT, Relation.PREVIOUS]:
                        time_window = np.random.randint(1, n_timepoints // 2)
                        decisions.append(Decision(feature, relation, threshold, time_window=time_window))
                    elif relation in [Relation.BEFORE, Relation.AFTER]:
                        time_window = np.random.randint(1, n_timepoints)
                        decisions.append(Decision(feature, relation, threshold, time_window=time_window))
                    else:
                        decisions.append(Decision(feature, relation, threshold))
        
        return decisions

    def _apply_decision(self, X, decision):
        n_samples, n_features, n_timepoints = X.shape
        if decision.relation == Relation.EXISTS:
            return (X[:, decision.feature] >= decision.threshold).any(axis=1)
        elif decision.relation == Relation.FORALL:
            return (X[:, decision.feature] >= decision.threshold).all(axis=1)
        elif decision.relation == Relation.EVENTUALLY:
            return np.array([np.any(X[i, decision.feature, j:] >= decision.threshold) for i in range(n_samples) for j in range(n_timepoints)]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.ALWAYS:
            return np.array([np.all(X[i, decision.feature, j:] >= decision.threshold) for i in range(n_samples) for j in range(n_timepoints)]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.UNTIL:
            return np.array([
                np.any((X[i, decision.feature, :j] >= decision.threshold) & (X[i, decision.secondary_feature, j] >= decision.secondary_threshold))
                for i in range(n_samples) for j in range(n_timepoints)
            ]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.NEXT:
            return np.array([
                np.any((X[i, decision.feature, j] >= decision.threshold) & (X[i, decision.feature, j+decision.time_window] >= decision.threshold))
                for i in range(n_samples) for j in range(n_timepoints-decision.time_window)
            ]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.PREVIOUS:
            return np.array([
                np.any((X[i, decision.feature, j] >= decision.threshold) & (X[i, decision.feature, j-decision.time_window] >= decision.threshold))
                for i in range(n_samples) for j in range(decision.time_window, n_timepoints)
            ]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.SINCE:
            return np.array([
                np.any((X[i, decision.secondary_feature, :j] >= decision.secondary_threshold) & np.all(X[i, decision.feature, j:] >= decision.threshold))
                for i in range(n_samples) for j in range(n_timepoints)
            ]).reshape(n_samples, -1).any(axis=1)
        elif decision.relation == Relation.BEFORE:
            return np.array([X[i, decision.feature, :decision.time_window] >= decision.threshold for i in range(n_samples)]).any(axis=1)
        elif decision.relation == Relation.AFTER:
            return np.array([X[i, decision.feature, decision.time_window:] >= decision.threshold for i in range(n_samples)]).any(axis=1)

    def _predict_single(self, x, node):
        if isinstance(node, LeafNode):
            return node.value
        
        if self._apply_decision(x.reshape(1, self.n_features_in_, self.n_timepoints_), node.decision)[0]:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            check_is_fitted(self)
            print("Modal Decision Tree Structure:")
            self.print_tree(self.tree_, 0)
            return

        indent = "  " * depth
        if isinstance(node, LeafNode):
            print(f"{indent}Leaf: prediction = {node.value}")
        else:
            print(f"{indent}Decision: {node.decision}")
            print(f"{indent}Left branch:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Right branch:")
            self.print_tree(node.right, depth + 1)

    def export_graphviz(self, filename="modal_decision_tree.dot"):
        check_is_fitted(self)
        
        def write_node(node, parent_id=None, branch=""):
            nonlocal node_id
            current_id = node_id
            node_id += 1

            if isinstance(node, LeafNode):
                dot_file.write(f'    node{current_id} [label="Leaf\\nValue: {node.value}", shape=box];\n')
            else:
                dot_file.write(f'    node{current_id} [label="{node.decision}"];\n')
                write_node(node.left, current_id, "yes")
                write_node(node.right, current_id, "no")

            if parent_id is not None:
                dot_file.write(f'    node{parent_id} -> node{current_id} [label="{branch}"];\n')

        with open(filename, "w") as dot_file:
            dot_file.write("digraph ModalDecisionTree {\n")
            node_id = 0
            write_node(self.tree_)
            dot_file.write("}")

        print(f"Decision tree exported to {filename}")
        print("To visualize, use: dot -Tpng modal_decision_tree.dot -o modal_decision_tree.png")

class ModalDecisionTreeRegressor(ModalDecisionTreeBase, RegressorMixin):
    def _leaf_value(self, y):
        return np.mean(y)

    def _evaluate_split(self, X, y, decision):
        mask = self._apply_decision(X, decision)
        
        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            return float('inf')
        
        left_mse = np.mean((y[mask] - np.mean(y[mask]))**2)
        right_mse = np.mean((y[~mask] - np.mean(y[~mask]))**2)
        return (np.sum(mask) * left_mse + np.sum(~mask) * right_mse) / len(y)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, allow_nd=True)
        if X.ndim != 3 or X.shape[1] != self.n_features_in_ or X.shape[2] != self.n_timepoints_:
            raise ValueError(f"X should have shape (n_samples, {self.n_features_in_}, {self.n_timepoints_})")
        
        return np.array([self._predict_single(x, self.tree_) for x in X])

class ModalDecisionTreeClassifier(ModalDecisionTreeBase, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return super().fit(X, y)

    def _leaf_value(self, y):
        return np.argmax(np.bincount(y))

    def _evaluate_split(self, X, y, decision):
        mask = self._apply_decision(X, decision)
        
        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            return float('inf')
        
        left_gini = 1 - sum((np.sum(y[mask] == c) / np.sum(mask))**2 for c in self.classes_)
        right_gini = 1 - sum((np.sum(y[~mask] == c) / np.sum(~mask))**2 for c in self.classes_)
        return (np.sum(mask) * left_gini + np.sum(~mask) * right_gini) / len(y)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, allow_nd=True)
        if X.ndim != 3 or X.shape[1] != self.n_features_in_ or X.shape[2] != self.n_timepoints_:
            raise ValueError(f"X should have shape (n_samples, {self.n_features_in_}, {self.n_timepoints_})")
        
        return np.array([self.classes_[self._predict_single(x, self.tree_)] for x in X])

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, allow_nd=True)
        if X.ndim != 3 or X.shape[1] != self.n_features_in_ or X.shape[2] != self.n_timepoints_:
            raise ValueError(f"X should have shape (n_samples, {self.n_features_in_}, {self.n_timepoints_})")
        
        predictions = np.array([self._predict_single(x, self.tree_) for x in X])
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0
        return proba