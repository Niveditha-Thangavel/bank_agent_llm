from crewai import Agent, Task, Crew,LLM
from crewai.tools import BaseTool
import json
import os
import re
from typing import Any


os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 
#API_KEY = "AIzaSyCG3JZiOqvWYmLrGHC9RoSbdnN4OkJVUgo"
API_KEY = "AIzaSyD0UGwiFALa_h-b20cIpWoErnd02oq2weI"
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
        goal=f"Extract the customer id from '{customer_id}' and fetch the exact customer data using the customer id ",
        backstory=f"You are an information fetcher that accurately retrieves all the "
                  f"financial records for customer id present in  '{customer_id}' ",
        tools = [fetch_tool()],
        allow_delegation = True,
        llm = ollm

    )

    approval_agent = Agent(
            role="Loan Decision Agent",
            goal=(
                "1) Transform raw financial records into structured metrics "
                "(e.g., average monthly income, average monthly spend, income volatility, "
                "recurring payments, transaction patterns like freq of credits/debits, "
                "large outliers). "
                "2) Apply scoring rules on those metrics, normalize the score between 0 and 10, "
                "and return loan approval status with concise, actionable reasons."
            ),
            backstory=(
                "You are both a fast, precise calculator and the loan manager: convert raw bank/transaction "
                "statements into scoring metrics and then apply lending rules to produce a normalized "
                "credit score (0-10) and an approval decision with reasons. If any metric is unclear, "
                "state assumptions explicitly."
            ),
            allow_delegation= False,
            llm=ollm
        )
    
    answer_agent = Agent(
        role = "Final answer provider",
        goal = "Extract the decision and reason from the output of the approval_agent and provide it as a dictionary",
        backstory = "You are an extractor that only fetches the decision and reason from the output of the approval agent",
        allow_delegation = False, 
        llm = ollm
    )
            
    return input_agent,approval_agent, answer_agent




def create_task(customer_id):
    input_agent,approval_agent, answer_agent = create_agents(customer_id)
    input_agent_task = Task(
        description=f"Fetch bank statements, credit card details, and loan data for customer {customer_id}.",
        expected_output="A JSON object containing all financial data for the customer.",
        agent=input_agent
    )

    approval_agent_task = Task(
    description="""
Process the input payload and compute metrics using the exact formulas specified below, then make an approval decision.
MANDATES:
- Fill "customer_id" from the input payload (look for input["customer_id"] or input["profile"]["customer_id"]).
- Return ONLY the exact JSON schema described in expected_output (no extra keys, no comments, no explanatory text).
- Use system date for account_age_months calculation.
- All numeric fields must be numbers; counts must be integers. Use math.inf for infinite ratios where required.

METRICS (exact formulas)
1) Income & Spend Metrics
- total_credit_amount = sum(amount for transaction in transactions if transaction["type"] == "credit")
- number_of_credit_transactions = count of such credit transactions (integer)
- total_debit_amount = sum(amount for transaction in transactions if transaction["type"] == "debit")
- number_of_debit_transactions = count of such debit transactions (integer)
- total_transactions = number_of_credit_transactions + number_of_debit_transactions
- avg_monthly_income = total_credit_amount / number_of_credit_transactions   (if number_of_credit_transactions == 0 -> 0.0)
- avg_monthly_spend  = total_debit_amount / number_of_debit_transactions    (if number_of_debit_transactions == 0 -> 0.0)
- total_transaction_amount = total_credit_amount + total_debit_amount
- avg_transaction_amount = total_transaction_amount / total_transactions   (if total_transactions == 0 -> 0.0)
- spend_ratio = (avg_monthly_spend / avg_monthly_income) * 100            (if avg_monthly_income == 0 -> set spend_ratio = math.inf)

2) Variance & Anomaly
- variance = mean((avg_transaction_amount - t_amount)**2 for each transaction)
  (if no transactions -> 0.0)
- standard_deviation = sqrt(variance)
- anomaly_flags = count of transactions where transaction_amount > standard_deviation

3) Late payments (credits_loan.json)
- late_payment_count = count of billing cycles where (payment_date > cycle_end) OR (amount_paid < amount_due)
  (if credits_loan data absent -> 0)

4) Current active loans
- current_loans = count of loans where outstanding_amount > 0  (if loan list absent -> 0)

5) Credit utilization ratio
- For each card in credit_cards:
    utilization_percent = (current_balance / credit_limit) * 100  (if credit_limit == 0 -> treat utilization_percent = 100)
- credit_utilization_ratio = average(utilization_percent values) (if no cards -> 0.0)

6) Account age
- account_age_months = whole months between today and account_creation_date (round down). If account_creation_date missing -> 0

7) Transaction pattern
- incoming_to_outgoing_ratio = total_credit_amount / total_debit_amount
    (if total_debit_amount == 0 and total_credit_amount > 0 -> math.inf;
     if both zero -> 0.0)
- transaction_pattern.anomaly_flags = anomaly_flags

DECISION RULES (apply after metrics)
- income_spend_ratio = avg_monthly_income / (avg_monthly_spend or 1e-9)   (use 1e-9 to avoid div-by-zero)
- REJECT if ANY:
    late_payment_count > 3
    credit_utilization_ratio > 70
    anomaly_flags > 0
    avg_month_income < avg_month_spend
    account_age_months < 3
    transaction_pattern.anomaly_flags > 0
- APPROVE if ALL:
    income_spend_ratio >= 1.5
    late_payment_count <= 1
    credit_utilization_ratio <= 40
    account_age_months >= 12
    anomaly_flags == 0
    transaction_pattern.incoming_to_outgoing_ratio >= 1.2   (treat inf as >1.2)
- REVIEW if ALL:
    1.1 <= income_spend_ratio < 1.5
    late_payment_count <= 2
    credit_utilization_ratio <= 60
    anomaly_flags == 0
- If none of the above match exactly, pick the most conservative decision among applicable states (REJECT > REVIEW > APPROVE). Provide failing checks in reason.

OUTPUT (EXACT JSON - no extra keys)
{
  "customer_id": string,
  "decision": "APPROVE" | "REJECT" | "REVIEW",
  "reason": string,
  "income_spend_ratio": number,
  "metrics": {
    "avg_month_income": number,
    "avg_month_spend": number,
    "total_transaction_amount": number,
    "avg_transaction_amount": number,
    "spend_ratio": number,
    "variance": number,
    "standard_deviation": number,
    "anomaly_flags": integer,
    "late_payment_count": integer,
    "current_loans": integer,
    "credit_utilization_ratio": number,
    "account_age_months": integer,
    "transaction_pattern": {
      "incoming_to_outgoing_ratio": number,
      "anomaly_flags": integer
    }
  },
  "metrics_notes": [ "strings describing parsing assumptions or missing fields" ]
}

Use input payload fields: transactions, credits_loan (if present), loans (if present), credit_cards (if present), account_creation_date, customer_id.
""",
    expected_output="""
Return EXACT JSON (no extra keys). Example structure:

{
  "customer_id": "{customer_id} ",
  "decision": "APPROVE" | "REJECT" | "REVIEW",
  "reason": "human readable explanation for the decision (include failing checks)",
  "income_spend_ratio": number,
  "metrics": {
    "avg_month_income": number,
    "avg_month_spend": number,
    "total_transaction_amount": number,
    "avg_transaction_amount": number,
    "spend_ratio": number,
    "variance": number,
    "standard_deviation": number,
    "anomaly_flags": integer,
    "late_payment_count": integer,
    "current_loans": integer,
    "credit_utilization_ratio": number,
    "account_age_months": integer,
    "transaction_pattern": {
      "incoming_to_outgoing_ratio": number,
      "anomaly_flags": integer
    }
  },
  "metrics_notes": [
    "optional list of parsing/assumption notes (strings)"
  ]
}
""",
    agent=approval_agent,
    context=[input_agent_task] 
)

    answer_agent_task = Task(
        description = "Fetch only the decision and reason from teh output of the approval agent and pass it as a dictionary",
        expected_output = "a dictionary containin the decision and the reason ",
        agent = answer_agent, 
        context = [approval_agent_task]
    )
    return input_agent_task, approval_agent_task, answer_agent_task




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
        input_agent, approval_agent , answer_agent = create_agents(customer_id)
        input_agent_task, approval_agent_task, answer_agent_task = create_task(customer_id)


        crew = Crew(
            agents=[input_agent,approval_agent, answer_agent],
            tasks=[input_agent_task, approval_agent_task, answer_agent_task],
            verbose=True
        )

        result = crew.kickoff()
        ans = extract_result(result)
        return ans

