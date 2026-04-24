import streamlit as st

from hr_langgraph_agent import get_agent_reply


st.set_page_config(page_title="HR Management Agent", page_icon="👩‍💼", layout="centered")
st.title("HR Management Agent")
st.caption("Ask about employee details, leave balances, leave requests, and HR policies.")

EMPLOYEE_OPTIONS = {
    "E101 - Aarav Mehta": "E101",
    "E102 - Sara Khan": "E102",
    "E103 - Dev Patel": "E103",
}

with st.sidebar:
    st.header("Employee Context")
    selected_employee_label = st.selectbox(
        "Select employee",
        list(EMPLOYEE_OPTIONS.keys()),
        index=0,
    )
    selected_employee_id = EMPLOYEE_OPTIONS[selected_employee_label]
    st.caption(f"Selected employee ID: `{selected_employee_id}`")

    st.divider()
    st.subheader("What can I help with?")
    st.markdown(
        """
        - Get employee details by ID
        - Check leave balance
        - Submit leave requests
        - Explain HR policies (leave, WFH, reimbursement)
        - Draft follow-up questions when information is missing
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns(2)
with col1:
    if st.button("Employee details"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"Get employee details for {selected_employee_id}",
            }
        )
with col2:
    if st.button("Leave balance"):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"Check leave balance for {selected_employee_id}",
            }
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_reply = get_agent_reply(st.session_state.messages[-1]["content"])
            except Exception as exc:
                assistant_reply = f"Error: {exc}"
            st.markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

user_prompt = st.chat_input("Type your HR question here...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.rerun()
