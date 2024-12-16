# CustomRandomForestClassifier

`CustomRandomForestClassifier` is a subclass of `sklearn.ensemble.RandomForestClassifier` that extends its functionality with additional features for analyzing and modifying decision trees within the forest.

## Features:
- **Modify Splits**: Change the split value of a specific node in a decision tree.
- **Access Trees**: Retrieve the individual decision trees within the random forest.
- **Extract and Print Split Information**: Extract split values for each feature and print the structure of the trees.

## Usage Example:

```python
from custom_rf import CustomRandomForestClassifier

# Initialize the classifier
clf = CustomRandomForestClassifier(n_estimators=10)

# Fit the model
clf.fit(X_train, y_train)

# Access a specific tree and modify its split value
clf.modify_split(tree_index=0, node_index=5, new_split_value=2.5)

# Extract splits for each feature
splits = clf.extract_splits()
