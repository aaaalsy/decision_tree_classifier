
-> Overview\
This project implements a Decision Tree Classifier to predict customer behavior in a banking context. Specifically, it predicts whether a client will subscribe to a term deposit (`y`) based on various demographic and marketing attributes.\
\
-> Machine Learning Pipeline\
* Preprocessing: Categorical variables are transformed using Label Encoding.\
* Modeling: A `DecisionTreeClassifier` is trained with a maximum depth of 5 to ensure interpretability and prevent overfitting.\
* Evaluation:\
    * Accuracy: Measured on a 20% test split.\
    * Cross-Validation: 5-fold CV is used to ensure model consistency.\
    * Metrics: Includes a full Classification Report and Confusion Matrix.\
\
-> Visualizations\
* The Decision Tree: A visual representation of the first 3 levels of the model's logic.\
* Feature Importance: A chart ranking the top 10 attributes that most influence the prediction (e.g., `duration`, `balance`).\
\
-> Requirements\
* Scikit-Learn, Pandas, Matplotlib}
