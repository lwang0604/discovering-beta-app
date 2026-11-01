import streamlit as st


pages = [
    st.Page("pages/Discovering_Beta.py", title="Discovering Beta", icon="📈", default=True),
    st.Page("pages/Diversification_Challenge.py", title="Diversification Challenge", icon="🧺"),
]

navigation = st.navigation(pages)
navigation.run()
