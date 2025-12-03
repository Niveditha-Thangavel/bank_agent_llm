from crewai import Agent, Task, Crew,LLM
from crewai.tools import BaseTool
import json
import os
import re
from typing import Any

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 
#API_KEY = "AIzaSyCG3JZiOqvWYmLrGHC9RoSbdnN4OkJVUgo"
API_KEY = "AIzaSyDuV74BfOJni44FGQI_2T3o9ratKaBZq0A"
ollm = LLM(model='gemini/gemini-2.5-flash', api_key=API_KEY)


class fetch_tool(BaseTool):
    name:str = "Data fetcher"
    description:str = "Fetches the right customer account data, from database(json file)"
    def _run(self,customer_id:str):
        with open ("bank_statements.json","r") as f:
            statement = json.load(f)

        with open ("credits_loan.json","r") as f1:
            credit_loan = json.load(f1)

        statement_customer = -1
        account_creation = -1
        credit = -1
        loans = -1

        for id in statement["bank_statements"]:
            if id["customer_id"] == customer_id:
                statement_customer = id["transactions"]

        for id in credit_loan["customer_accounts"]:
            if id["customer_id"] == customer_id:
                account_creation = id["account_creation_date"]
                credit = id["credit_cards"]
                loans = id["loans"]
        data = {"transactions": statement_customer, "account_creation_date":account_creation,"credit_card":credit, "loans":loans}
        return data


def create_agents(customer_id):
    input_agent = Agent(
        role="Input Agent",
        goal=f"Retrieve complete financial records and available metadata for customer '{customer_id}'.",
        backstory="You fetch bank statements, credit card records, loans, and available account metadata. Provide the raw payload for decision making.",
        tools=[fetch_tool()],
        allow_delegation=True,
        llm=ollm
    )

    decision_agent = Agent(
        role="Decision Agent",
        goal=(
            "Label each rule using one of three states: Passing State, Borderline State, or Breaking State. "
            "Map the collection of labeled rules to a final decision using the provided mapping table. "
            "Return ONLY a JSON object with the keys 'decision' and 'reason' and no other keys or text."
        ),
        backstory=(
            "You evaluate rule states (Passing / Borderline / Breaking) for each named rule and produce the final decision."
        ),
        allow_delegation=False,
        llm=ollm
    )

    return input_agent, decision_agent


def create_task(customer_id):
    input_agent, decision_agent = create_agents(customer_id)

    input_agent_task = Task(
        description=f"Fetch the full financial payload for customer {customer_id}.",
        expected_output="Raw JSON financial payload.",
        agent=input_agent
    )

    decision_agent_task = Task(
        description="""
Evaluate the customer's financial profile by assigning states to the following rules (no numeric computations shown here).
Use the rule names exactly as written. Provide the final result as JSON with only 'decision' and 'reason'.

Rules:
1. Income Check: Income must be ≥ ₹20,000 per month
2. Account Age: Account must be ≥ 6 months old
3. Payment History: Late payments must be ≤ 2
4. Transaction Issues: There must be no transaction anomalies
5. Credit Usage: Credit utilization must be < 70%
6. Current Loans: Customer must have ≤ 1 active loan
7. Income–Spend Health Check: Monthly income must show a clear positive margin over monthly spending
8. Transaction Activity Check: Customer should have consistent and healthy transaction activity
9. Outlier Behavior Check: There must be no extreme or unexplained large transaction outliers
10. Liquidity Buffer Check: Customer should maintain a reasonable financial buffer or savings room
11. Credit History Strength: Customer must show reliable and stable historical credit behavior
12. Documentation & Identity Check: Customer must have complete and verifiable documentation & identity records

the loan is approved if all the rules are satisfied
the loan is set to review id 8 or more rules are satisfied but not all
the loan is rejected if the rules satisfied are below 8

""",
        expected_output='{"decision":"APPROVE|REVIEW|REJECT","reason":"string"}',
        agent=decision_agent,
        context=[input_agent_task]
    )

    return input_agent_task, decision_agent_task

def handle_prompt(prompt: str) -> Any:
    prompt = prompt or ""
    valid = re.search(r"\b[Cc]\d{3}\b", prompt)
    if valid:
        return valid.group(0).upper()
    near = re.search(r"\b[A-Za-z]\d+\b", prompt)
    if near:
        return "INVALID"
    return None


def extract_result(result):
            # 1. If crew produced a proper dict
            if hasattr(result, "json_dict") and isinstance(result.json_dict, dict):
                return result.json_dict
            
            # 2. If raw JSON is inside result.raw
            if hasattr(result, "raw") and isinstance(result.raw, str):
                try:
                    return json.loads(result.raw)
                except:
                    pass

            # 3. Check tasks_output → usually has final JSON
            if hasattr(result, "tasks_output"):
                for t in result.tasks_output:
                    if hasattr(t, "raw"):
                        raw = t.raw
                        if isinstance(raw, dict):
                            return raw
                        if isinstance(raw, str):
                            try:
                                return json.loads(raw)
                            except:
                                pass

            raise ValueError("Could not extract JSON from CrewOutput")

def main(user_prompt):  
    result = handle_prompt(user_prompt)

    if result is None:
        print("Customer ID missing — please provide your ID (e.g., C101).")
    elif result == "INVALID":
        print("Invalid customer ID. Please provide an ID in the format C101 (C + 3 digits).")
    else:
        customer_id = result
        input_agent, decision_agent = create_agents(customer_id)
        input_agent_task, decision_agent_task = create_task(customer_id)


        crew = Crew(
            agents=[input_agent,decision_agent],
            tasks=[input_agent_task, decision_agent_task],
            verbose=True
        )

        result = crew.kickoff()
        print(result)
