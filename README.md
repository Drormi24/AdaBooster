# AdaBooster
Goal: Implement AdaBoost algorithm on a Weak Learning decision Tree.

Dataset: Heart Disease UCI
Dataset Creators:
Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
Donor: David W. Aha (aha '@' ics.uci.edu)

Steps: 
1. Build a weak learner using a decision tree.
2. Initialize the data points weight w(i) to be equal for all points.
3. Calcualte the error for each data point.
4. Calcualte dataset total error.
5. Calculate the significance.
6. Update weigths of the points.
7. Build the weighted sum of weak learners to produce one strong learner 
   using the significance, and return the sign of the result.

Results and conclusions:
Decision Tree Model accuracy is:  81.25 %
Logistic Reg. Model accuracy is:  88.02 %
SVM Model accuracy is:  88.02 %
Model accuracy is:  88.54 %
This code demonstrates the power of boosting and using a combination of various ML tools.
Enjoy
