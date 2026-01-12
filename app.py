import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Page config (branding)
# ---------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Simple CSS (card + spacing)
# ---------------------------
st.markdown("""
<style>
/* Reduce top padding a bit */
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}

/* Nice title */
h1 {letter-spacing: -0.5px;}

/* Card style */
.card {
    padding: 1.1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}

/* Subtle helper text */
.helper {opacity: 0.8; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model + encoders (cache to avoid reloading)
# ---------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("extra_trees_credit_modelo.pkl")
    encoders = {
        col: joblib.load(f"{col}_encoder.pkl")
        for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
    }
    return model, encoders

model, encoders = load_assets()

# ---------------------------
# Sidebar (profile + links)
# ---------------------------
with st.sidebar:
    st.markdown("## 👋 Fernando Valencia")
    st.markdown('<div class="helper">Industrial Engineer • BI / Data Science</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("- 💼 [LinkedIn](https://www.linkedin.com/in/fernando-valencia-ds/)")
    st.markdown("- 🧠 [GitHub](https://github.com/FernandoValencia-DS)")
    st.markdown("---")
    with st.expander("ℹ️ About this app"):
        st.write(
            "Credit risk prediction app using a Machine Learning model." \
            "Enter the applicant’s data and get a prediction (GOOD/BAD)."
        )

# ---------------------------
# Main header
# ---------------------------
st.title("💳 Credit Risk Prediction")
st.write("Enter applicant information to predict level of risk")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Applicant information")

    with st.form("risk_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1,help="Employment level of the applicant (see details below).")
            housing = st.selectbox("Housing", ["own", "rent", "free"])
            

        with c2:
            saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"], help="Level of the applicant’s savings (see details below).")
            checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"], help="Balance level of the checking account (see details below).")
            credit_amount = st.number_input("Credit Amount", min_value=1, value=1000, step=50)
            duration = st.number_input("Duration (months)", min_value=1, value=12)
           
        with st.expander("❓ Field definitions"):
            st.markdown("""
            ### 💼 Job (0–3)
            - **0**: Unskilled / unemployed  
            - **1**: Unskilled but resident  
            - **2**: Skilled employee  
            - **3**: Highly skilled / management  

            ### 💰 Saving Accounts
            Indicates the **level of savings** available to the applicant.
            - **little**: Very low or no savings  
            - **moderate**: Some savings, but limited  
            - **rich**: High level of savings  
            - **quite rich**: Very high savings, strong financial buffer  

            ### 🏦 Checking Account
            Represents the **balance level of the checking account** used for daily transactions.
            - **little**: Low balance  
            - **moderate**: Average balance  
            - **rich**: High balance  
            """)

        submitted = st.form_submit_button("Predict Risk")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")
    st.markdown('<div class="helper">Prediction and Metrics.</div>', unsafe_allow_html=True)

    if submitted:
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration],
        })

        pred = model.predict(input_df)[0]

        # If model supports probabilities, show them
        proba_bad = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            # OJO: aquí asumimos que la clase "1" es GOOD como en tu lógica original.
            # Ajusta el índice según tu entrenamiento.
            # Si pred==1 => GOOD, entonces "BAD" sería la otra clase.
            proba_good = float(proba[list(model.classes_).index(1)]) if 1 in model.classes_ else None
            if proba_good is not None:
                proba_bad = 1 - proba_good

        # Display
        if pred == 1:
            st.success("Predicted credit risk: **GOOD**")
        else:
            st.error("Predicted credit risk: **BAD**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Credit Amount", f"{credit_amount:,.0f}")
        m2.metric("Duration (months)", f"{duration}")
        if proba_bad is not None:
            m3.metric("Estimated GOOD probability", f"{proba_good*100:.1f}%")
            st.progress(min(max(proba_good, 0.0), 1.0))

        with st.expander("🔎 See encoded input"):
            st.dataframe(input_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
