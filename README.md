# CS540-Decision-Tree-
CS540(AI) This program builds a classification decision tree for categorical attributes, including building a tree from a training dataset and pruning the learned decision tree with a tuning dataset. The DecisionTreeImpl.java is implemented by Han Jiang, and other files are provided by the instructor Collin.
The DecisionTree is implemented in the DecisionTreeImp.java, However, I do not consider the situation that two subtrees have same accuracy when pruning. My solution is to choose the first one when doing DFS, and actually there are many other methods, such as choosing the smaller one. 