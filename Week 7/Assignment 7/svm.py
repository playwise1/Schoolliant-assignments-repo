import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# (a) One-hot encoding dummy data
df_tiny = pd.DataFrame({
    "limit_balance": [20000, 120000, 90000, 50000],
    "marriage": [1, 2, 3, 2],
    "default": [1, 0, 0, 1]
})
df_encoded = pd.get_dummies(df_tiny, columns=["marriage"])
print(df_encoded)
# Explanation: Using pd.get_dummies prevents the model from assuming category 3 is numerically closer to 2 than to 1.

# (b) Load, Split, Scale, Baseline SVM
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_base = SVC().fit(X_train_scaled, y_train)
ConfusionMatrixDisplay.from_estimator(svm_base, X_test_scaled, y_test)
plt.show()

# (c) GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best Params:", grid.best_params_)
print("Tuned Test Accuracy:", grid.score(X_test_scaled, y_test))

# (d) Compare baseline and tuned. The accuracy gain is typically small, matching the concept that SVMs are strong "out of the box".

# (e) PCA Bonus
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

plt.plot(range(1, 3), pca.explained_variance_ratio_, marker='o')
plt.title("Scree Plot")
plt.show()

svm_pca = SVC(**grid.best_params_).fit(X_train_pca, y_train)
# Plotting decision boundary logic involves generating a meshgrid, predicting with svm_pca, and using plt.contourf.