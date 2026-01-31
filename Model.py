import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from concurrent.futures import ProcessPoolExecutor
import warnings
import os
import joblib


warnings.filterwarnings("ignore")

output_folder = '/home/centos/S/hcm/jupyter/Al Alloy/xiaolunwen/Fig-2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =========================

# =========================
df = pd.read_excel(
    '/home/centos/S/hcm/jupyter/Al Alloy/temperature_0.xlsx',
    engine='openpyxl'
)

X = df.drop(['序号', 'Tensile strength/MPa'], axis=1)
y = df['Tensile strength/MPa']

# =========================
# =========================
corr_matrix = X.corr().abs()
threshold = 0.7
columns_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > threshold:
            columns_to_drop.add(corr_matrix.columns[i])

X_reduced = X.drop(columns=columns_to_drop)

# =========================
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# =========================
# =========================
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

feature_importance_df = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importance_df.head(15)
print("Top features:\n", top_features['Feature'].tolist())

# =========================
# =========================
def evaluate_combination(combination):
    selected_features = list(combination) + ['Al']
    X_train_subset = X_train[selected_features]

    rf_model = RandomForestRegressor(random_state=42)
    cv_scores = cross_val_score(
        rf_model,
        X_train_subset,
        y_train,
        cv=10,
        scoring='r2'
    )

    return {
        'Features': selected_features,
        'Mean CV R2': np.mean(cv_scores)
    }

all_combinations = []
for r in range(0, 15):
    combos = itertools.combinations(
        top_features['Feature'].tolist(),
        r
    )
    all_combinations.extend(list(combos))

with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(executor.map(evaluate_combination, all_combinations))

# =========================
# =========================
results_df = pd.DataFrame(results)
results_df['Num Features'] = results_df['Features'].apply(len)

group_max = results_df.groupby('Num Features')['Mean CV R2'].max().reset_index()

best_idx = results_df['Mean CV R2'].idxmax()
best_score = results_df.loc[best_idx, 'Mean CV R2']
best_dim = results_df.loc[best_idx, 'Num Features']
best_features = results_df.loc[best_idx, 'Features']

print(f"\nThe best is: {best_features}")
print(f"R² = {best_score:.4f}")

# =========================
# =========================
plt.figure(figsize=(16, 13), dpi=600)

plt.scatter(
    results_df['Num Features'],
    results_df['Mean CV R2'],
    marker='s',
    facecolors='none',
    edgecolors='navy',
    s=360,
    linewidths=0.8,
    zorder=1
)

plt.scatter(
    group_max['Num Features'],
    group_max['Mean CV R2'],
    color='#F4A582',
    s=450,
    zorder=3
)

plt.scatter(
    best_dim,
    best_score,
    color='red',
    marker='*',
    s=1100,
    zorder=4,
    label='Highest'
)

plt.xlabel('Number of exhaustive features', fontsize=32)
plt.ylabel('R²(10-Fold CV)', fontsize=32)
plt.xticks(range(0, 16), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(0, 16)
plt.ylim(0.8, 0.90)

plt.legend(frameon=False, fontsize=28)
plt.grid(False)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, 'Figure_i_exhaustive_search.png'),
    dpi=600,
    bbox_inches='tight'
)
plt.show()

# =========================
# =========================
best_model = RandomForestRegressor(random_state=42)
X_best = X[list(best_features)]
best_model.fit(X_best, y)

y_pred_train = best_model.predict(X_train[list(best_features)])
y_pred_test = best_model.predict(X_test[list(best_features)])

print("Train R²:", r2_score(y_train, y_pred_train))
print("Test  R²:", r2_score(y_test, y_pred_test))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Test  RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

joblib.dump(
    best_model,
    os.path.join(output_folder, 'best_random_forest_model.pkl')
)

selected_data = df[list(best_features) + ['Tensile strength/MPa']]
selected_data.to_excel(
    os.path.join(output_folder, 'selected_features_and_target.xlsx'),
    index=False
)
