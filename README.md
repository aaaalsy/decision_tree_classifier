{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0  \
\
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