import streamlit as st

from agent import HolidayAgent

# Set web page title and icon.
st.set_page_config(
    page_title="Best Holiday Plan 💬 ",
    page_icon=":robot:"
)

user_input = st.text_input("Ask a question about your holiday plan:")

if user_input:

    my_agent = HolidayAgent()

    result = my_agent.run(user_input)

    st.write(result)
    


