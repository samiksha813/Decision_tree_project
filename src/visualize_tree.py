import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree as sktree
from sklearn.tree import plot_tree

# Import our implementation
from decision_tree import DecisionTreeClassifier, DecisionNode

def plot_decision_boundaries(clf, X, y, feature_names, target_names):
    """Plot decision boundaries of the decision tree."""
    # Create a grid of points to plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Make predictions on the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='black')
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundaries")
    
    # Create a legend
    legend_labels = [f"{target_names[i]}" for i in range(len(target_names))]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    
    plt.show()

def visualize_tree_structure(node, feature_names, target_names, depth=0, prefix=""):
    """Recursively print the tree structure."""
    if node.is_leaf_node():
        print(f"{prefix}└── Predict: {target_names[node.value]}")
        return
    
    print(f"{prefix}├── {feature_names[node.feature_idx]} <= {node.threshold:.2f}")
    if node.left:
        visualize_tree_structure(node.left, feature_names, target_names, depth + 1, prefix + "│   ")
    
    print(f"{prefix}└── {feature_names[node.feature_idx]} > {node.threshold:.2f}")
    if node.right:
        visualize_tree_structure(node.right, feature_names, target_names, depth + 1, prefix + "    ")

def main():
    # Load data
    iris = load_iris()
    X = iris.data[:, :2]  # Use only the first two features for visualization
    y = iris.target
    feature_names = iris.feature_names[:2]
    target_names = iris.target_names
    
    # Train the decision tree
    clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
    clf.fit(X, y)
    
    # Plot decision boundaries
    plot_decision_boundaries(clf, X, y, feature_names, target_names)
    
    # Print tree structure
    print("\nDecision Tree Structure:")
    visualize_tree_structure(clf.root, feature_names, target_names)
    
    # Compare with scikit-learn's implementation for verification
    print("\nFor comparison, here's scikit-learn's decision tree structure:")
    sk_clf = sktree.DecisionTreeClassifier(max_depth=3, criterion='gini')
    sk_clf.fit(X, y)
    
    plt.figure(figsize=(15, 10))
    plot_tree(sk_clf, 
              feature_names=feature_names,
              class_names=target_names,
              filled=True, 
              rounded=True)
    plt.title("Scikit-learn's Decision Tree")
    plt.show()

if __name__ == "__main__":
    main()
