import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_regression
import warnings

warnings.filterwarnings('ignore')

# 1. Load the modified data
try:
    df = pd.read_csv('cars.csv')
    print(f"Dataset Loaded! Shape: {df.shape}")
    print("Top features check (Correlation):")
    print(df[['Price', 'Safety_Rating', 'HP', 'Speed']].corr()['Price'].sort_values(ascending=False))
    print("-" * 50)
except FileNotFoundError:
    print("Error: 'cars.csv' not found.")
    exit()

target = 'Price'
X = df.drop(columns=[target])
y = df[target]

# 2. Preprocessing
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include=['object']).columns:
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

X_filled = X_encoded.fillna(X_encoded.mean())
y_binned = pd.qcut(y, q=3, labels=[0, 1, 2]) # Binned target for Chi2/Fisher

# ==========================================================
# INDIVIDUAL METHODS
# ==========================================================

# Method 1: Missing Value Ratio
print("\n--- 1. MISSING VALUE RATIO (USEFULNESS) ---")
missing_ratio = (df.drop(columns=[target]).isnull().sum() / len(df)) * 100
usefulness = 100 - missing_ratio
mv_df = pd.DataFrame({'Feature': X.columns, 'Usefulness': usefulness.values}).sort_values('Usefulness', ascending=False)
print(mv_df.to_string(index=False))

# Method 2: Information Gain
print("\n--- 2. INFORMATION GAIN ---")
ig_scores = mutual_info_regression(X_filled, y, random_state=42)
ig_df = pd.DataFrame({'Feature': X.columns, 'Score': ig_scores}).sort_values('Score', ascending=False)
print(ig_df.to_string(index=False))

# Method 3: Chi-Square
print("\n--- 3. CHI-SQUARE TEST ---")
X_pos = X_filled - X_filled.min()
chi2_scores, _ = chi2(X_pos, y_binned)
chi2_df = pd.DataFrame({'Feature': X.columns, 'Score': chi2_scores}).sort_values('Score', ascending=False)
print(chi2_df.to_string(index=False))

# Method 4: Fisher's Score
print("\n--- 4. FISHER'S SCORE ---")
def get_fisher(X, y):
    scores = []
    for col in X.columns:
        m = X[col].mean()
        num = sum([len(X[y==c]) * (X[col][y==c].mean() - m)**2 for c in np.unique(y)])
        den = sum([len(X[y==c]) * X[col][y==c].var() for c in np.unique(y)])
        scores.append(num / den if den > 0 else 0)
    return np.array(scores)

fisher_scores = get_fisher(X_filled, y_binned)
fisher_df = pd.DataFrame({'Feature': X.columns, 'Score': fisher_scores}).sort_values('Score', ascending=False)
print(fisher_df.to_string(index=False))

# ==========================================================
# FINAL COMPARISON & VISUALIZATION
# ==========================================================
print("\n" + "="*80)
print("FINAL CONSENSUS (All Scores Normalized 0-100)")
print("="*80)

comparison = pd.DataFrame({'Feature': X.columns})
def normalize(data):
    return (data - data.min()) / (data.max() - data.min()) * 100

comparison['Info_Gain'] = normalize(ig_scores)
comparison['Chi2'] = normalize(chi2_scores)
comparison['Fisher'] = normalize(fisher_scores)
comparison['Usefulness'] = usefulness.values

comparison['Average'] = comparison[['Info_Gain', 'Chi2', 'Fisher']].mean(axis=1)
comparison = comparison.sort_values('Average', ascending=False)

print(comparison[['Feature', 'Info_Gain', 'Chi2', 'Fisher', 'Usefulness', 'Average']].to_string(index=False))

# Visualization
plt.figure(figsize=(10, 5))
plt.barh(comparison['Feature'][::-1], comparison['Average'][::-1], color='crimson')
plt.title('Feature Priority for Price Prediction (Safety-First Data)')
plt.xlabel('Consensus Score (0-100)')
plt.tight_layout()
plt.savefig('safety_priority_selection.png')
plt.show()