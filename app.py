# import streamlit as st
# import joblib
# import numpy as np
# # Load the trained regression model
# model=joblib.load('regression_model.joblib')
# # App UI
# st.title("Job Package Prediction Based on CGPA")
# st.write("Enter your CGPA to predict the expected job pacakge:")

# # CGPA input
# cgpa = st.number_input(
#     "CGPA",
#     min_value=0.0,
#     max_value=10.0,
#     step=0.1
# )
# #Predict button
# if st.button("Predict Package"):
#     # Prepare input for model
#     input_data=np.array([[cgpa]])

#     # Make prediction
#     prediction=model.predict(input_data)

#     # Convert NumPy output to Python flpat safely
#     predicted_value=prediction.item()

#     # Optional:Prevent nagative output
#     predicted_value=max(predicted_value,0)

#     # Dispaly result
#     st.success(f"Predicted package: ₹{predicted_value:,.2f}LPA")


# import streamlit as st
# import joblib
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# # ------------------ Page Config ------------------
# st.set_page_config(
#     page_title="Job Package Predictor",
#     page_icon="💼",
#     layout="centered"
# )

# # ------------------ Load Model ------------------
# @st.cache_resource
# def load_model():
#     return joblib.load("regression_model.joblib")

# model = load_model()

# # ------------------ Sidebar ------------------
# st.sidebar.header("⚙️ Settings")
# show_chart = st.sidebar.checkbox("Show CGPA vs Package Chart", value=True)
# animate = st.sidebar.checkbox("Enable Prediction Animation", value=True)

# st.sidebar.markdown("---")
# st.sidebar.info("📌 Prediction is an estimate based on past data.")

# # ------------------ Main UI ------------------
# st.title("💼 Job Package Predictor")
# st.markdown(
#     "Predict your **expected job package (LPA)** based on your **CGPA** 🎓"
# )

# # CGPA Input (Slider + Box)
# cgpa = st.slider("Select your CGPA", 0.0, 10.0, 7.0, 0.1)
# cgpa = st.number_input("Or type CGPA manually", 0.0, 10.0, cgpa, 0.1)

# # ------------------ Predict ------------------
# if st.button("🚀 Predict Package"):
#     input_data = np.array([[cgpa]])

#     if animate:
#         with st.spinner("Analyzing CGPA..."):
#             time.sleep(1.5)

#     prediction = model.predict(input_data)
#     predicted_value = max(float(prediction.item()), 0)

#     # Result
#     st.success(f"🎯 **Predicted Package:** ₹{predicted_value:,.2f} LPA")

#     # Fun reaction
#     if predicted_value >= 10:
#         st.balloons()
#     elif predicted_value >= 6:
#         st.toast("🔥 Solid package!", icon="💪")
#     else:
#         st.toast("📈 Keep pushing, you’ll get there!", icon="🚀")

    # ----------
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load the trained regression model
model = joblib.load('regression_model.joblib')

# App UI
st.set_page_config(page_title="CGPA vs Job Package", layout="centered")

st.title("🎓 Job Package Prediction Based on CGPA")
st.write("Enter your CGPA to predict the expected job package and visualize it on the graph.")

# CGPA input
cgpa = st.slider(
    "Select your CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

# Generate CGPA range for graph
cgpa_range = np.linspace(0, 10, 100).reshape(-1, 1)
predicted_packages = model.predict(cgpa_range)
predicted_packages = np.maximum(predicted_packages, 0)

# Predict for selected CGPA
input_data = np.array([[cgpa]])
prediction = model.predict(input_data).item()
prediction = max(prediction, 0)

# Plotly interactive graph
fig = go.Figure()

# Regression line
fig.add_trace(go.Scatter(
    x=cgpa_range.flatten(),
    y=predicted_packages,
    mode='lines',
    name='Predicted Package Trend',
))

# User CGPA point
fig.add_trace(go.Scatter(
    x=[cgpa],
    y=[prediction],
    mode='markers',
    name='Your CGPA',
    marker=dict(size=12, symbol='circle')
))

fig.update_layout(
    title="📈 CGPA vs Predicted Job Package",
    xaxis_title="CGPA",
    yaxis_title="Package (LPA)",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# Display prediction
st.success(f"💼 Predicted Package: ₹{prediction:,.2f} LPA")
