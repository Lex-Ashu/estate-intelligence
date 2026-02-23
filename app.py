"""
🏠 Estate Intelligence — Real Estate Property Price Prediction App
Built with Streamlit · Pandas · NumPy · Matplotlib · Joblib

This app predicts housing prices using property features.
Currently uses a placeholder formula until the trained model (.pkl) is available.
"""

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
model = None  # will hold the loaded model if available

try:
    import joblib
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)  # TODO: swap with real model
except Exception as e:
    model = None  # graceful fallback — we'll warn the user in the sidebar


# ──────────────────────────────────────────────────────────────
# HELPER — PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────
def predict_price(area, bedrooms, bathrooms, stories, mainroad,
                  guestroom, basement, hotwaterheating,
                  airconditioning, parking, prefarea,
                  furnishingstatus):
    """
    Returns predicted price.
    Uses the trained model if available; otherwise falls back to a
    hand-tuned placeholder formula.
    """
    features = np.array([[area, bedrooms, bathrooms, stories,
                          mainroad, guestroom, basement,
                          hotwaterheating, airconditioning,
                          parking, prefarea, furnishingstatus]])

    if model is not None:
        # TODO: swap with real model
        return model.predict(features)[0]
    else:
        # TODO: swap with real model
        price = (area * 480
                 + bedrooms * 45000
                 + bathrooms * 35000
                 + stories * 20000
                 + airconditioning * 80000
                 + prefarea * 60000)
        return price


# ──────────────────────────────────────────────────────────────
# HELPER — FORMAT PRICE IN INDIAN ₹ (lakh / crore style)
# ──────────────────────────────────────────────────────────────
def format_inr(amount):
    """Format a number into Indian Rupee style with commas.
    Example: 4500000 → ₹45,00,000"""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    # last three digits
    result = s[-3:]
    s = s[:-3]
    # then groups of two
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return f"₹{result}"


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS — Navy / Dark-Blue Professional Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Main background ─── */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #112240 50%, #0a1628 100%);
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #162a50 100%);
        border-right: 1px solid rgba(100,160,255,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #cdd9e5 !important;
    }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 8px 8px 0 0;
        color: #8babc7;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(100,160,255,0.12) !important;
        color: #64a0ff !important;
        border-bottom: 2px solid #64a0ff;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #1a5aff, #0d3fb8);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(26,90,255,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3373ff, #1a5aff);
        box-shadow: 0 6px 20px rgba(26,90,255,0.45);
        transform: translateY(-1px);
    }

    /* ─── Metric / Price Box ─── */
    .price-box {
        background: linear-gradient(135deg, #0d3fb8 0%, #1a5aff 100%);
        border: 1px solid rgba(100,160,255,0.25);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 20px auto;
        max-width: 480px;
        box-shadow: 0 8px 32px rgba(13,63,184,0.4);
    }
    .price-box .label {
        color: #a3c4f3;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .price-box .price {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* ─── Headers ─── */
    h1, h2, h3 {
        color: #e2eaf3 !important;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0;
    }
    .sub-title {
        color: #8babc7 !important;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ─── Inputs ─── */
    .stSelectbox label, .stSlider label, .stRadio label,
    .stFileUploader label {
        color: #cdd9e5 !important;
        font-weight: 500;
    }

    /* ─── Status badges ─── */
    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-ready {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .status-estimate {
        background: rgba(250,204,21,0.15);
        color: #facc15;
        border: 1px solid rgba(250,204,21,0.3);
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        color: #5a7a9a;
        font-size: 0.8rem;
        margin-top: 50px;
        padding: 16px 0;
        border-top: 1px solid rgba(100,160,255,0.1);
    }

    /* ─── Radio buttons horizontal ─── */
    .stRadio > div {
        flex-direction: row;
        gap: 12px;
    }

    /* ─── DataFrame styling ─── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Estate Intelligence")
    st.markdown(
        "AI-powered real estate price prediction tool. "
        "Enter property details to get an instant price estimate."
    )

    st.markdown("---")

    # Team name placeholder
    st.markdown("**Team:** _Your Team Name Here_")

    st.markdown("---")

    # ── Model status indicator ──
    st.markdown("### 📦 Model Status")
    if model is not None:
        st.markdown(
            '<span class="status-badge status-ready">● Model Ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-estimate">● Using Estimate Mode</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "No trained model found at `models/best_model.pkl`. "
            "Using a placeholder formula for estimates.",
            icon="ℹ️",
        )

    st.markdown("---")

    # ── Instructions ──
    st.markdown("### 📖 How to Use")
    st.markdown(
        """
        1. **Single Prediction** — Fill in property details in Tab 1 
           and click *Predict Price*.
        2. **Batch Prediction** — Upload a CSV in Tab 2 to predict 
           prices for multiple properties at once.
        3. **Download** — Export batch results as CSV for further 
           analysis.
        """
    )


# ──────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏠 Estate Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predict residential property prices with machine learning</p>',
    unsafe_allow_html=True,
)
st.markdown("")

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏡 Single Property Prediction", "📊 Batch CSV Prediction"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SINGLE PROPERTY PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### Enter Property Details")

    # Two-column layout for inputs
    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("📐 Area (sq ft)", min_value=500, max_value=10000,
                          value=1500, step=50)
        bedrooms = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4], index=0)
        stories = st.selectbox("🏢 Stories", options=[1, 2, 3, 4], index=0)
        parking = st.selectbox("🚗 Parking Spots", options=[0, 1, 2, 3], index=0)
        furnishing = st.selectbox(
            "🪑 Furnishing Status",
            options=["Unfurnished", "Semi-Furnished", "Furnished"],
            index=0,
        )

    with col2:
        mainroad = st.radio("🛣️ Main Road Access", ["Yes", "No"], index=0,
                            horizontal=True)
        guestroom = st.radio("🚪 Guest Room", ["Yes", "No"], index=1,
                             horizontal=True)
        basement = st.radio("🏗️ Basement", ["Yes", "No"], index=1,
                            horizontal=True)
        hotwaterheating = st.radio("♨️ Hot Water Heating", ["Yes", "No"], index=1,
                                   horizontal=True)
        airconditioning = st.radio("❄️ Air Conditioning", ["Yes", "No"], index=1,
                                    horizontal=True)
        prefarea = st.radio("⭐ Preferred Area", ["Yes", "No"], index=1,
                            horizontal=True)

    # ── Encode user inputs to numeric ──
    mainroad_val = 1 if mainroad == "Yes" else 0
    guestroom_val = 1 if guestroom == "Yes" else 0
    basement_val = 1 if basement == "Yes" else 0
    hotwaterheating_val = 1 if hotwaterheating == "Yes" else 0
    airconditioning_val = 1 if airconditioning == "Yes" else 0
    prefarea_val = 1 if prefarea == "Yes" else 0
    furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
    furnishing_val = furnishing_map[furnishing]

    st.markdown("")

    # ── Predict button ──
    if st.button("🔮 Predict Price", use_container_width=True, key="single_predict"):
        predicted = predict_price(
            area, bedrooms, bathrooms, stories,
            mainroad_val, guestroom_val, basement_val,
            hotwaterheating_val, airconditioning_val,
            parking, prefarea_val, furnishing_val,
        )

        # ── Styled price output ──
        st.markdown(
            f"""
            <div class="price-box">
                <div class="label">Estimated Property Price</div>
                <div class="price">{format_inr(predicted)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Feature importance bar chart (dummy values) ──
        st.markdown("#### 📊 Feature Importance")
        st.caption("Shows how much each feature contributes to the prediction.")

        # TODO: swap with real model importances
        importance_data = {
            "Area": 0.45,
            "Bathrooms": 0.15,
            "Air Conditioning": 0.12,
            "Preferred Area": 0.10,
            "Bedrooms": 0.10,
            "Stories": 0.08,
        }

        features_sorted = dict(sorted(importance_data.items(),
                                       key=lambda x: x[1]))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#0a1628")
        ax.set_facecolor("#0a1628")

        bars = ax.barh(
            list(features_sorted.keys()),
            list(features_sorted.values()),
            color=["#1a5aff", "#2d6bff", "#4080ff", "#5c96ff",
                   "#7eadff", "#a3c4f3"],
            edgecolor="none",
            height=0.55,
            zorder=3,
        )
        ax.set_xlim(0, 0.55)
        ax.set_xlabel("Relative Importance", color="#8babc7",
                       fontsize=10, fontweight=500)
        ax.tick_params(colors="#8babc7", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#1e3a5f")
        ax.spines["left"].set_color("#1e3a5f")
        ax.grid(axis="x", color="#1e3a5f", linewidth=0.5, zorder=0)

        # value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{width:.0%}", va="center", color="#a3c4f3",
                    fontsize=9, fontweight=600)

        plt.tight_layout()
        st.pyplot(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — BATCH CSV PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### Upload a CSV File")
    st.caption(
        "Your CSV should contain these columns: "
        "`area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, "
        "`basement`, `hotwaterheating`, `airconditioning`, `parking`, "
        "`prefarea`, `furnishingstatus`"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read & display the uploaded data
        df = pd.read_csv(uploaded_file)
        st.markdown("#### 📄 Uploaded Data")
        st.dataframe(df, use_container_width=True)

        # Required columns check
        required_cols = [
            "area", "bedrooms", "bathrooms", "stories", "mainroad",
            "guestroom", "basement", "hotwaterheating", "airconditioning",
            "parking", "prefarea", "furnishingstatus",
        ]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(
                f"❌ Missing columns in your CSV: **{', '.join(missing)}**. "
                "Please make sure all required columns are present."
            )
        else:
            if st.button("🚀 Predict All", use_container_width=True,
                         key="batch_predict"):
                # Apply prediction row-wise
                df["predicted_price"] = df.apply(
                    lambda row: predict_price(
                        row["area"], row["bedrooms"], row["bathrooms"],
                        row["stories"], row["mainroad"], row["guestroom"],
                        row["basement"], row["hotwaterheating"],
                        row["airconditioning"], row["parking"],
                        row["prefarea"], row["furnishingstatus"],
                    ),
                    axis=1,
                )

                # Format for display
                df["predicted_price_formatted"] = df["predicted_price"].apply(
                    format_inr
                )

                st.markdown("#### ✅ Predictions Complete")
                st.dataframe(df, use_container_width=True)

                # Download button
                csv_output = df.drop(columns=["predicted_price_formatted"]).to_csv(
                    index=False
                )
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_output,
                    file_name="estate_intelligence_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "⚠️ Predictions are estimates only. Not financial advice. "
    "| Built with ❤️ using Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
