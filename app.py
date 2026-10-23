"""
Streamlit chat interface for the literature review assistant.

Run with:
    streamlit run app.py
"""
import streamlit as st

from query import answer

st.set_page_config(page_title="Literature Review Assistant", layout="wide")
st.title("Literature Review Assistant")
st.caption("Answers come only from your papers. Always check cited passages before using them.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about your papers:")

if st.button("Ask") and question.strip():
    with st.spinner("Searching your papers and generating an answer..."):
        text, hits = answer(question.strip())
    st.session_state.history.insert(0, {"q": question.strip(), "a": text, "hits": hits})

for item in st.session_state.history:
    st.markdown(f"### {item['q']}")
    st.markdown(item["a"])
    with st.expander("Retrieved passages"):
        for i, h in enumerate(item["hits"], start=1):
            st.markdown(f"**[{i}] {h['source']}, p. {h['page']}**")
            st.write(h["text"])
            st.divider()
    st.divider()
