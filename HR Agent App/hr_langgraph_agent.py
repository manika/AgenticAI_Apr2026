import os
from typing import Dict, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

load_dotenv()


# Demo in-memory employee dataset for HR workflows.
EMPLOYEES: Dict[str, Dict[str, str]] = {
    "E101": {"name": "Aarav Mehta", "department": "Engineering", "manager": "Riya Shah"},
    "E102": {"name": "Sara Khan", "department": "Human Resources", "manager": "Neha Verma"},
    "E103": {"name": "Dev Patel", "department": "Finance", "manager": "Arjun Rao"},
}

LEAVE_BALANCE: Dict[str, int] = {"E101": 14, "E102": 20, "E103": 8}

POLICIES: Dict[str, str] = {
    "leave": "Employees can take up to 24 paid leaves per year. Planned leaves should be applied 3 days in advance.",
    "work from home": "Employees can work from home up to 2 days per week with manager approval.",
    "reimbursement": "Travel and training reimbursements must be submitted within 30 days of expense date.",
}


@tool
def get_employee_details(employee_id: str) -> str:
    """Fetch basic employee details using employee ID (for example: E101)."""
    employee = EMPLOYEES.get(employee_id.upper())
    if not employee:
        return f"No employee found for ID {employee_id}."
    return (
        f"Employee ID: {employee_id.upper()}\n"
        f"Name: {employee['name']}\n"
        f"Department: {employee['department']}\n"
        f"Manager: {employee['manager']}"
    )


@tool
def check_leave_balance(employee_id: str) -> str:
    """Check available leave balance for an employee ID."""
    employee_id = employee_id.upper()
    if employee_id not in LEAVE_BALANCE:
        return f"No leave balance record found for {employee_id}."
    return f"{employee_id} has {LEAVE_BALANCE[employee_id]} leave days available."


@tool
def submit_leave_request(employee_id: str, days: int, reason: str) -> str:
    """Submit a leave request and update leave balance for demo usage."""
    employee_id = employee_id.upper()
    if employee_id not in LEAVE_BALANCE:
        return f"Cannot submit request. No employee record for {employee_id}."
    if days <= 0:
        return "Leave days must be greater than zero."
    if LEAVE_BALANCE[employee_id] < days:
        return (
            f"Leave request rejected. {employee_id} has only "
            f"{LEAVE_BALANCE[employee_id]} day(s) available."
        )

    LEAVE_BALANCE[employee_id] -= days
    return (
        f"Leave request submitted for {employee_id}: {days} day(s), reason: {reason}. "
        f"Updated leave balance: {LEAVE_BALANCE[employee_id]} day(s)."
    )


@tool
def get_hr_policy(topic: str) -> str:
    """Get an HR policy summary by topic (leave, work from home, reimbursement)."""
    topic_key = topic.strip().lower()
    for key, value in POLICIES.items():
        if key in topic_key or topic_key in key:
            return f"Policy for {key}: {value}"
    return (
        "Policy not found for that topic. Available topics are: "
        + ", ".join(sorted(POLICIES.keys()))
    )


def build_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export your API key before running this script."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools: List = [
        get_employee_details,
        check_leave_balance,
        submit_leave_request,
        get_hr_policy,
    ]
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are an HR management assistant. Use tools for employee details, leave balances, "
            "leave requests, and policy retrieval. Ask follow-up questions if the user misses "
            "critical information like employee ID or leave days."
        ),
    )


def get_agent_reply(user_input: str) -> str:
    """Generate a single response for a user query."""
    agent = build_agent()
    response = agent.invoke({"messages": [("user", user_input)]})
    return response["messages"][-1].content


def run_chat():
    print("HR Management Agent is ready. Type 'exit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print(f"\nHR Agent: {get_agent_reply(user_input)}")


if __name__ == "__main__":
    run_chat()
