import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer & Default Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (glassmorphism, clean dark theme elements, premium fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #1E293B;
    }
    
    /* Professional Solid Slate Title */
    .title-gradient {
        color: #0F172A;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.1);
        border-color: rgba(37, 99, 235, 0.4);
    }
    
    /* Prediction Badges */
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        text-align: center;
    }
    .badge-default {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .badge-nodefault {
        background-color: #D1FAE5;
        color: #059669;
        border: 1px solid #6EE7B7;
    }
    
    /* Solid line divider */
    .gradient-line {
        height: 1px;
        background: #E2E8F0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_model():
    model_path = 'Scikit-Learn/xgboost_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# Helper function to load data
@st.cache_data
def load_dataset():
    # Prefer original data for EDA because it has clean categorical labels (Own, Rent, Education, etc.)
    original_path = 'Original Data/credit_risk.csv'
    if os.path.exists(original_path):
        return pd.read_csv(original_path)
    data_path = 'Data Cleaned/credit_risk_final.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model = load_model()
df_clean = load_dataset()

# Title Section
st.markdown('<div class="title-gradient">Credit Risk & Default Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Evaluate applicant default risk and explore machine learning model performance using a calibrated XGBoost Classifier.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_predict, tab_performance, tab_eda = st.tabs([
    "Predict Loan Default",
    "Model Performance & Insights",
    "Exploratory Data Analysis"
])

# ==========================================
# TAB 1: PREDICT LOAN DEFAULT
# ==========================================
with tab_predict:
    st.markdown("### Enter Applicant Details")
    st.markdown("Use the controls below to configure the applicant's financial and demographic profile to predict their default risk.")
    
    # Create two columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Personal Profile")
        age = st.slider("Applicant Age", min_value=18, max_value=85, value=28, step=1)
        income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=55000, step=1000, format="%d")
        emp_length = st.slider("Employment Length (Years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
        home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col2:
        st.markdown("##### Loan Details")
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=500, max_value=500000, value=12000, step=500, format="%d")
        interest_rate = st.slider("Interest Rate (%)", min_value=3.0, max_value=25.0, value=11.5, step=0.1)
        loan_intent = st.selectbox(
            "Loan Purpose / Intent",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
        )

    with col3:
        st.markdown("##### Credit History")
        cred_length = st.slider("Credit History Length (Years)", min_value=0, max_value=30, value=5, step=1)
        past_default = st.selectbox("Has a History of Past Default?", ["No", "Yes"])
        
        # Calculate percent of income dynamically
        pct_income = loan_amount / income if income > 0 else 0.0
        st.markdown(f"<p style='font-size: 14px; color: #64748B;'>Percent of Income: <strong>{pct_income * 100:.2f}%</strong></p>", unsafe_allow_html=True)
        if pct_income > 1.0:
            st.warning("Warning: Requested loan amount exceeds applicant's annual income.")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("Calculate Default Risk", use_container_width=True):
        if model is None:
            st.error("Error: Model file 'Scikit-Learn/xgboost_model.pkl' not found. Please train the model first.")
        else:
            # Preprocess inputs
            # Log transform skewed inputs: Income, Emp_length, Amount, Age
            # Note: np.log1p is used in model training
            log_age = np.log1p(age)
            log_income = np.log1p(income)
            log_emp_length = np.log1p(emp_length)
            log_amount = np.log1p(loan_amount)
            
            # Map default history
            binary_default = 1 if past_default == "Yes" else 0
            
            # One-hot encoding mapping
            input_dict = {
                'Age': log_age,
                'Income': log_income,
                'Emp_length': log_emp_length,
                'Amount': log_amount,
                'Rate': interest_rate,
                'Percent_income': pct_income,
                'Default': binary_default,
                'Cred_length': cred_length,
                'Home_MORTGAGE': home_ownership == "MORTGAGE",
                'Home_OTHER': home_ownership == "OTHER",
                'Home_OWN': home_ownership == "OWN",
                'Home_RENT': home_ownership == "RENT",
                'Intent_DEBTCONSOLIDATION': loan_intent == "DEBTCONSOLIDATION",
                'Intent_EDUCATION': loan_intent == "EDUCATION",
                'Intent_HOMEIMPROVEMENT': loan_intent == "HOMEIMPROVEMENT",
                'Intent_MEDICAL': loan_intent == "MEDICAL",
                'Intent_PERSONAL': loan_intent == "PERSONAL",
                'Intent_VENTURE': loan_intent == "VENTURE"
            }
            
            # Construct DataFrame
            df_input = pd.DataFrame([input_dict])
            
            # Make prediction
            prob = model.predict_proba(df_input)[0][1]
            pred = model.predict(df_input)[0]
            
            # Create columns to display result
            res_col1, res_col2 = st.columns([1, 1.5])
            
            with res_col1:
                st.markdown("### Prediction Result")
                if pred == 1:
                    st.markdown('<div class="badge badge-default">HIGH RISK OF DEFAULT</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 1rem; font-size: 18px;'>The model predicts a default probability of <strong>{prob * 100:.2f}%</strong>. It is highly recommended to inspect credit history details or reject/restructure the loan application.</p>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-nodefault">APPROVED / LOW RISK</div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top: 1rem; font-size: 18px;'>The model predicts a default probability of <strong>{prob * 100:.2f}%</strong>. The loan application meets the standard credit safety requirements.</p>", unsafe_allow_html=True)
                    
            with res_col2:
                # Gauge plot for probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Risk Probability (%)", 'font': {'size': 20, 'family': 'Outfit'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#6366F1"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#E2E8F0",
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 60], 'color': '#FEF3C7'},
                            {'range': [60, 100], 'color': '#FEE2E2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: MODEL PERFORMANCE & INSIGHTS
# ==========================================
with tab_performance:
    st.markdown("### Model Performance Evaluation")
    st.markdown("This tab displays evaluation metrics comparing Scikit-Learn algorithms alongside the final PyCaret training runs.")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("##### Scikit-Learn Model Benchmark (Tuned)")
        # Create tabular dataframe matching final_results_status.csv
        benchmarks = {
            "Model": ["XGBoost", "Gradient Boosting", "Random Forest", "Support Vector Machines", "K-Nearest Neighbors", "Naive Bayes"],
            "Accuracy": [0.9218, 0.9205, 0.9150, 0.8555, 0.8401, 0.8082],
            "Precision": [0.9491, 0.9640, 0.9487, 0.8528, 0.7893, 0.5756],
            "Recall": [0.6969, 0.6796, 0.6662, 0.4486, 0.4152, 0.6302],
            "F1-Score": [0.8037, 0.7972, 0.7827, 0.5879, 0.5442, 0.6017],
            "AUC-ROC": [0.8429, 0.8360, 0.8277, 0.7127, 0.6911, 0.7458]
        }
        df_bench = pd.DataFrame(benchmarks)
        st.dataframe(df_bench.style.highlight_max(axis=0, subset=df_bench.columns[1:], color="#D1FAE5"), hide_index=True, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: #64748B;'>Values correspond to evaluations conducted on the holdout test set (10%).</p>", unsafe_allow_html=True)
        
    with m_col2:
        st.markdown("##### PyCaret Automated Model Search (CV Results)")
        pycaret_bench = {
            "Model": ["Extreme Gradient Boosting (xgboost)", "Light Gradient Boosting (lightgbm)", "Random Forest (rf)", "CatBoost (catboost)", "Gradient Boosting (gbc)", "Extra Trees (et)"],
            "Accuracy": [0.9320, 0.9271, 0.9180, 0.9178, 0.9137, 0.9031],
            "AUC-ROC": [0.9534, 0.9460, 0.9256, 0.9304, 0.9196, 0.9058],
            "Recall": [0.7357, 0.6961, 0.6691, 0.6647, 0.6517, 0.6223],
            "Precision": [0.9395, 0.9586, 0.9368, 0.9411, 0.9322, 0.9039],
            "F1-Score": [0.8251, 0.8065, 0.7806, 0.7791, 0.7670, 0.7370]
        }
        df_pycaret = pd.DataFrame(pycaret_bench)
        st.dataframe(df_pycaret.style.highlight_max(axis=0, subset=df_pycaret.columns[1:], color="#D1FAE5"), hide_index=True, use_container_width=True)
        st.markdown("<p style='font-size: 12px; color: #64748B;'>Values correspond to K-Fold cross-validation runs performed inside the PyCaret pipeline.</p>", unsafe_allow_html=True)
        
    st.markdown("<div class='gradient-line'></div>", unsafe_allow_html=True)
    
    st.markdown("##### Feature Importance")
    st.markdown("Below are the factors influencing model decisions. Model features are extracted from the trained XGBoost model.")
    
    if model is not None:
        importances = model.feature_importances_
        feature_names = model.feature_names_in_
        
        df_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(
            df_importances, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            color='Importance',
            color_continuous_scale=px.colors.sequential.Blues,
            title="Feature Importance - Calibrated XGBoost",
            labels={'Importance': 'Importance Weight', 'Feature': 'Dataset Feature'}
        )
        fig_imp.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance will be displayed after model training.")

    st.markdown("<div class='gradient-line'></div>", unsafe_allow_html=True)

    # Images Section (Confusion Matrix, ROC Curve)
    st.markdown("##### Model Diagnostics Plots")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        st.markdown("**Confusion Matrix**")
        cm_path = "Scikit-Learn/Confusion Matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Confusion Matrix image not found.")
            
    with p_col2:
        st.markdown("**ROC Curve**")
        roc_path = "Scikit-Learn/ROC Curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)
        else:
            st.info("ROC Curve image not found.")
            
    with p_col3:
        st.markdown("**Precision Recall Curve**")
        pr_path = "Scikit-Learn/Precision Recall.png"
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
        else:
            st.info("Precision-Recall curve image not found.")

# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab_eda:
    st.markdown("### Interactive Data Explorer")
    st.markdown("Explore historical customer distributions and default relationships inside the loan book database.")
    
    if df_clean is not None:
        # Check if we have categorical columns
        has_categorical = 'Home' in df_clean.columns and 'Intent' in df_clean.columns
        
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            # Distribution of income by home ownership
            fig_eda1 = px.box(
                df_clean,
                x='Home' if has_categorical else ('Home_RENT' if 'Home_RENT' in df_clean.columns else df_clean.columns[10]),
                y='Income',
                color='Status',
                title="Income Distribution by Home Ownership & Default Status",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Status': 'Default Status (0=Paid, 1=Default)', 'Home': 'Home Ownership'}
            )
            fig_eda1.update_yaxes(type="log")
            fig_eda1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eda1, use_container_width=True)
            
        with eda_col2:
            # Scatter plot of loan amount vs income
            fig_eda2 = px.scatter(
                df_clean.head(8000), # Sample first 8k rows for rendering speed
                x='Income',
                y='Amount',
                color='Status',
                opacity=0.6,
                title="Loan Amount vs Annual Income (Sampled)",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Status': 'Default Status'}
            )
            fig_eda2.update_xaxes(type="log")
            fig_eda2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eda2, use_container_width=True)
            
        st.markdown("<div class='gradient-line'></div>", unsafe_allow_html=True)
        
        # Histograms of client demographic details
        eda_col3, eda_col4 = st.columns(2)
        
        with eda_col3:
            fig_eda3 = px.histogram(
                df_clean,
                x='Age',
                color='Status',
                barmode='group',
                nbins=30,
                title="Age Distribution Profile of Borrowers",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Status': 'Default Status'}
            )
            fig_eda3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eda3, use_container_width=True)
            
        with eda_col4:
            fig_eda4 = px.histogram(
                df_clean,
                x='Rate',
                color='Status',
                barmode='group',
                title="Loan Interest Rate Distributions",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Status': 'Default Status'}
            )
            fig_eda4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eda4, use_container_width=True)
            
        if has_categorical:
            st.markdown("<div class='gradient-line'></div>", unsafe_allow_html=True)
            st.markdown("##### Loan Intent Analysis")
            # Show default rates and counts by loan intent
            fig_eda5 = px.histogram(
                df_clean,
                x='Intent',
                color='Status',
                barmode='group',
                title="Loan Purpose/Intent Frequency by Default Status",
                color_discrete_map={0: "#10B981", 1: "#EF4444"},
                labels={'Status': 'Default Status', 'Intent': 'Loan Purpose'}
            )
            fig_eda5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eda5, use_container_width=True)
    else:
        st.error("No dataset available to show in EDA tab. Please verify the dataset files in the project repository.")
