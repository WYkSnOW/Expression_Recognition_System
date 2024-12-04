import streamlit as st

proposal_page = st.Page(
    page="proposal.py",
    title="Proposal",
)
midterm_page = st.Page(
    page="midterm_report.py",
    title="Midterm Report"
)

final_page = st.Page(
    page="final_report.py",
    title="Final Report",
    default=True
)

demo_page = st.Page(
    page="demo.py",
    title="Demo"
)

pg = st.navigation(pages=[proposal_page, midterm_page, final_page, demo_page])

if demo_page.title == "Demo":
    st.session_state["st_page"] = "demo"
else:
    st.session_state["st_page"] = None

pg.run()
