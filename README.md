🚖 Uber H3 Spatial Surge & Demand Engine

An end-to-end spatial-temporal machine learning pipeline and interactive dashboard that forecasts rideshare demand density and predicts 15-minute advance surge pricing multipliers using Uber's H3 Spatial Indexing System, XGBoost, and Streamlit.


📌 Features & Architecture:

Uber H3 Geospatial Indexing: Maps raw latitude/longitude coordinates to Resolution 8 hexagonal cells (~0.73 \text{km}^2 area).
Spatial k-Ring Features: Captures localized demand spillover across adjacent neighboring hexagons.
Temporal Lags: Computes 15-minute and 60-minute demand lags to model time-series trends.
XGBoost Predictive Engine: Gradient boosting regressor trained on chronological splits to forecast surge multipliers.
3D PyDeck Visual Dashboard: Interactive Streamlit web app with dynamic demand shock simulation controls.
🛠️ Tech Stack
Languages & Libraries: Python, Uber H3 (h3), PyDeck, Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit.
