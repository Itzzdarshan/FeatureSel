#Feature Selection: Filter, Wrapper, and Exhaustive Methods

Feature selection is the process of isolating the most relevant input variables for your model. It reduces computational cost, improves model interpretability, and mitigates overfitting.

1. Filter Methods
Filter methods use statistical techniques to evaluate the relationship between each feature and the target variable, independent of any machine learning algorithm.

* Missing Value Filter: Removes columns where the null-value percentage exceeds a threshold (e.g., 40-50%).
* Mutual Information: Measures the "dependency" between variables. Unlike correlation, it captures non-linear relationships.
* Chi-Square Test ($\chi^2$): Used for categorical features to see if the occurrence of a specific feature and the target are independent.
* Fisher’s Score: Evaluates class separation. It seeks features with High Between-Class Variance and Low Within-Class Variance.

2. Wrapper Methods
Wrapper methods treat feature selection as a search problem. They use a specific ML model (e.g., Logistic Regression) as a "black box" to evaluate feature subsets.

* Forward Selection: Starts with zero features and adds the "best performer" iteratively.
* Backward Elimination: Starts with all features and removes the "weakest performer" at each step.
* Recursive Feature Elimination (RFE): A more robust approach that trains a model, ranks features by importance (weights), and prunes the least important ones recursively.

3. Exhaustive Feature Selection
The "Brute Force" approach of feature selection.
* Definition: It evaluates every possible combination of features (e.g., for 10 features, it tests 2^{10} combinations).
* Advantage: Guaranteed to find the absolute best-performing subset for a specific model.
* Disadvantage: Computationally prohibitive. For a dataset with 50 features, an exhaustive search is impossible due to the exponential growth of combinations.

4. Selection Strategy
* Use Filter Methods as a first pass to remove obviously irrelevant data or highly missing values.
* Use Wrapper/RFE when you have a specific model in mind and a manageable number of features.
* Use Exhaustive only for extremely small, high-stakes feature sets where every decimal of accuracy matters.
