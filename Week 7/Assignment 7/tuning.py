from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

wine = load_wine()
X, y = wine.data, wine.target

# (a) Square-root starting value: sqrt(13) = 3.6 -> roughly 3 or 4.

# (b) Loop max_features
max_features_options = [2, 3, 4, 5, 6, 8, 13]
oob_errors = []
for mf in max_features_options:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, oob_score=True, random_state=42)
    rf.fit(X, y)
    oob_errors.append(1 - rf.oob_score_)

# (c) Plot
plt.plot(max_features_options, oob_errors, marker='o')
plt.xlabel("Max Features")
plt.ylabel("OOB Error")
plt.show()
# The plot identifies the optimal configuration (lowest error). It typically aligns closely with the square root rule (~3 or 4).

# (d) Loop n_estimators
n_trees = [10, 50, 100, 300]
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X, y)
    print(f"Trees: {n}, OOB Error: {1 - rf.oob_score_:.3f}")
# As the number of trees grows, OOB error stabilizes and generally decreases up to a point.

# (e) Compare
best_rf = RandomForestClassifier(n_estimators=300, max_features=3, random_state=42)
rf_score = cross_val_score(best_rf, X, y, cv=5).mean()
dt_score = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5).mean()
print(f"Best RF CV Score: {rf_score:.3f} | Single DT CV Score: {dt_score:.3f}")