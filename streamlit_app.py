import streamlit as st

proposal_page = st.Page(
    page="proposal.py",
    title="Proposal",
)
midterm_page = st.Page(
    page="midterm_report.py",
    title="Midterm Report",
    default=True
)

final_page = st.Page(
    page="final_report.py",
    title="Final Report",
)

pg = st.navigation(pages=[proposal_page, midterm_page, final_page])

pg.run()