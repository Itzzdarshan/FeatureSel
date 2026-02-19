import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="FeatureIntel Pro", page_icon="🧬", layout="wide")

# --- LUXURY CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #050a18;
        color: #e0e0e0;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #050a18 !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
def fisher_score(X, y):
    scores = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        overall_mean = np.mean(feature)
        num, den = 0, 0
        for cls in np.unique(y):
            cls_feat = feature[y == cls]
            num += len(cls_feat) * (np.mean(cls_feat) - overall_mean) ** 2
            den += len(cls_feat) * np.var(cls_feat)
        scores.append(num / den if den != 0 else 0)
    return np.array(scores)

# --- HEADER ---
st.markdown('<div class="main-title">GENETIC FEATURE RADIOLOGY</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Statistical Filtering & Signal Analysis Suite</p>", unsafe_allow_html=True)

# --- DATA LOADING ---
try:
    df = pd.read_csv("iris_features.csv")
    X = df.drop('target', axis=1).values
    y = df['target'].values
    feature_names = df.columns[:-1]
except:
    st.error("Please run train.py first to generate the dataset!")
    st.stop()

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛠️ Analysis Control")
    method = st.selectbox("Select Filter Method", ["Information Gain", "Chi-Squared", "Fisher Score"])
    
    st.markdown("---")
    st.write("**Methodology Description:**")
    if method == "Information Gain":
        st.caption("Measures the reduction in entropy. Higher gain indicates high predictive power.")
        
    elif method == "Chi-Squared":
        st.caption("Tests the independence of features relative to the target class.")
    else:
        st.caption("Calculates the ratio of between-class variance to within-class variance.")
        
        
    execute = st.button("RUN STATISTICAL SCAN")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if execute:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Calculation
        if method == "Information Gain":
            scores = mutual_info_classif(X, y)
        elif method == "Chi-Squared":
            X_norm = MinMaxScaler().fit_transform(X)
            scores, _ = chi2(X_norm, y)
        else:
            scores = fisher_score(X, y)
            
        # Visualization
        results_df = pd.DataFrame({'Feature': feature_names, 'Importance': scores}).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        sns.barplot(data=results_df, x='Importance', y='Feature', palette="viridis", ax=ax)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        
        # Display Rankings
        st.markdown("### 🏆 Ranked Importance")
        for i, row in results_df.iterrows():
            st.write(f"**Rank {i+1}:** {row['Feature']} — `{row['Importance']:.4f}`")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Select a metric and press 'Run' to begin feature selection analysis.")

# --- MISSING VALUE SECTION ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📉 Integrity Check: Missing Value Ratio")
m_df = pd.read_csv("missing_sample.csv")
ratios = m_df.isnull().mean()

m_cols = st.columns(4)
for i, (col_name, val) in enumerate(ratios.items()):
    m_cols[i].metric(label=col_name, value=f"{val*100:.0f}%", delta="Missing Data", delta_color="inverse")
st.markdown('</div>', unsafe_allow_html=True)