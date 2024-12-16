class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, **kwargs):
        """
        Initializes the custom random forest classifier.
        """
        super().__init__(**kwargs)
        self.feature_names_ = None
    
    def fit(self, X, y, sample_weight=None):
        """
        Fits the model to the data, storing feature names if X is a DataFrame.
        """
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]
        super().fit(X, y, sample_weight)
    
    def predict(self, X):
        """
        Predicts class labels for the given data.
        """
        return super().predict(X)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for the given data.
        """
        return super().predict_proba(X)

    def get_params(self, deep=True):
        """
        Gets the parameters of the classifier.
        """
        return super().get_params(deep)

    def set_params(self, **params):
        """
        Sets the parameters of the classifier.
        """
        return super().set_params(**params)

    def get_trees(self):
        """
        Returns a list of decision trees in the random forest.
        """
        return self.estimators_

    def modify_split(self, tree_index, node_index, new_split_value):
        """
        Modifies the split value of a specific node in one of the trees.
        
        :param tree_index: Index of the tree to modify.
        :param node_index: Index of the node to modify.
        :param new_split_value: The new split value.
        """
        tree = self.estimators_[tree_index]
        tree_ = tree.tree_

        if node_index >= len(tree_.feature):
            raise ValueError("Invalid node index.")

        tree_.threshold[node_index] = new_split_value
    
    def print_tree_structure(self, tree_index):
        """
        Prints the structure of a decision tree, showing feature indices, split values, and decisions.
        
        :param tree_index: Index of the tree to print.
        """
        tree = self.estimators_[tree_index]
        tree_ = tree.tree_

        print("Tree Structure:")
        print(f"Feature indices: {tree_.feature}")
        print(f"Split values: {tree_.threshold}")
        print(f"Decision values: {tree_.value}")
        
    def print_splits(self, tree_index):
        """
        Prints all splits (feature indices and threshold values) of a specific decision tree.
        
        :param tree_index: Index of the tree to analyze.
        """
        tree = self.estimators_[tree_index]
        tree_ = tree.tree_
        
        for node_id in range(len(tree_.feature)):
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            if feature_idx != -2:
                print(f"Node {node_id}: Feature {feature_idx}, Split value {threshold}")
            else:
                print(f"Node {node_id}: Leaf node (no split)")
                
    def extract_splits(self):
        """
        Extracts the split values for each feature across all trees.
        
        :return: A dictionary with feature names as keys and lists of split values as values.
        """
        if self.feature_names_ is None:
            raise ValueError("The model has not been trained yet.")
        
        splits_dict = {name: [] for name in self.feature_names_}

        for tree in self.estimators_:
            tree_ = tree.tree_

            for node_id in range(len(tree_.feature)):
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                
                if feature_idx != -2:
                    feature_name = self.feature_names_[feature_idx]
                    splits_dict[feature_name].append(threshold)

        return splits_dict
