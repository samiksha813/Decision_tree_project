import numpy as np
from collections import Counter

class DecisionNode:
    """A decision node in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree (samples <= threshold)
        self.right = right              # Right subtree (samples > threshold)
        self.value = value              # Value if leaf node (class label)

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    """Decision Tree Classifier implementation from scratch."""
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _gini(self, y):
        """Calculate Gini impurity for a set of labels."""
        if len(y) == 0:
            return 0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(np.square(probabilities))
    
    def _entropy(self, y):
        """Calculate entropy for a set of labels."""
        if len(y) == 0:
            return 0
            
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calculate information gain for a given split."""
        if self.criterion == 'gini':
            parent_impurity = self._gini(y)
        else:  # entropy
            parent_impurity = self._entropy(y)
            
        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
            
        # Calculate weighted impurity of children
        n = len(y)
        n_left, n_right = len(y[left_mask]), len(y[right_mask])
        
        if self.criterion == 'gini':
            left_impurity = self._gini(y[left_mask])
            right_impurity = self._gini(y[right_mask])
        else:  # entropy
            left_impurity = self._entropy(y[left_mask])
            right_impurity = self._entropy(y[right_mask])
            
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain is the difference between parent and child impurities
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -1
        best_feature_idx, best_threshold = None, None
        
        n_samples, n_features = X.shape
        
        # Try all features and all possible thresholds
        for feature_idx in range(n_features):
            # Get unique feature values as potential thresholds
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        if feature_idx is None:  # No split found
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)
        
        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(feature_idx=feature_idx, 
                          threshold=threshold, 
                          left=left_subtree, 
                          right=right_subtree)
    
    def _most_common_label(self, y):
        """Find the most common label in a set of labels."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def fit(self, X, y):
        """Build the decision tree."""
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """Predict the class for a single sample."""
        if node.is_leaf_node():
            return node.value
            
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
