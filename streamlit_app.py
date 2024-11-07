import streamlit as st

proposal_page = st.Page(
    page="views/proposal.py",
    title="Proposal",
    default=True,
)
midterm_page = st.Page(
    page="views/midterm_report.py",
    title="Midterm Report",
)
final_page = st.Page(
    page="views/final_report.py",
    title="Final Report",
)

pg = st.navigation(pages=[proposal_page, midterm_page, final_page])

pg.run()