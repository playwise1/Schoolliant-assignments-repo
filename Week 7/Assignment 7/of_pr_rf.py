from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# (a) No limits
dt_full = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("No limit Train Acc:", dt_full.score(X_train, y_train))
print("No limit Test Acc:", dt_full.score(X_test, y_test))
# Observation: Train accuracy is 1.0 (overfitting), while test accuracy is lower.

# (b) Pruned tree
dt_pruned = DecisionTreeClassifier(min_samples_leaf=5, random_state=42).fit(X_train, y_train)
print("Pruned Train Acc:", dt_pruned.score(X_train, y_train))
print("Pruned Test Acc:", dt_pruned.score(X_test, y_test))
# Explanation: Train accuracy drops, but test accuracy improves because requiring a minimum samples per leaf prevents the model from fitting training noise.

# (c) Random Forest
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42).fit(X_train, y_train)
print("RF Test Acc:", rf.score(X_test, y_test))
print("OOB Score:", rf.oob_score_)
print("OOB Error:", 1 - rf.oob_score_)

# (d) Feature importances
importances = rf.feature_importances_
indices = importances.argsort()[-10:] # top 10
plt.barh([data.feature_names[i] for i in indices], importances[indices])
plt.title("Top 10 Feature Importances")
plt.show()