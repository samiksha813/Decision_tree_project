import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import our implementation
from decision_tree import DecisionTreeClassifier

def load_data():
    """Load and prepare the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names

def main():
    # Load data
    X, y, feature_names, target_names = load_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Initialize and train the decision tree
    print("\nTraining Decision Tree...")
    clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(5):
        print(f"Sample {i+1}: Predicted={target_names[y_pred[i]]}, Actual={target_names[y_test[i]]}")

if __name__ == "__main__":
    main()
