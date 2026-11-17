import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "attends_live": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    "submits_on_time": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "has_laptop": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    "passed": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
})

# (a) Build X and y, train classifier
X = df[["attends_live", "submits_on_time", "has_laptop"]]
y = df["passed"]
clf = DecisionTreeClassifier(criterion='gini', random_state=0)
clf.fit(X, y)

# (b) Display tree
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=X.columns, class_names=["Failed", "Passed"], filled=True)
plt.show()
# The root node is 'submits_on_time', matching the hand calculation.

# (c) Gini of root node matches the data (5 passed, 5 failed -> 1 - (0.5)^2 - (0.5)^2 = 0.5)
# (d) Predict: attends(1), submits(1), laptop(0). Hand trace: submits=1 (Left), then attends=1 (Left), laptop=0 (Right) -> leaf predicts 1 (Passed).
print("Prediction:", clf.predict([[1, 1, 0]])) 

# (e) The root node is "submits_on_time". Internal nodes are "attends_live" and "has_laptop". 
# The leaves are the terminal nodes predicting the final class. All leaves in this fully grown tree are pure (Gini = 0.0).