from __future__ import annotations

import streamlit as st

from viz.app.views.run_inspector import main


st.set_page_config(page_title="ADCS KalmanNet Run Inspector", layout="wide")
main()
