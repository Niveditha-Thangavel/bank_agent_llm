from crewai import Agent, Task, Crew,LLM
from crewai.tools import BaseTool
import json
import os


os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 
ollm = LLM(
    model="ollama/phi3",
    base_url="http://localhost:11434"

)






class fetch_tool(BaseTool):
    name:str = "Data fetcher"
    description:str = "Fetches the right customer account data, from database(json file)"
    def _run(self,customer_id:str):
        with open ("bank_statement.json","r") as f:
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
        verbose = True,
        llm = ollm
        

    )

    calculator_agent = Agent(
        role="Calculator Agent",
        goal="Transform raw financial records into structured metrics like "
             "average monthly income, average monthly spend, transaction patterns, etc., "
             "and forward them to the approval agent.",
        backstory="You are a fast, precise calculator who converts raw statements "
                  "into credit-scoring parameters.",
        allow_delegation = True,
        verbose = True,
        llm = ollm
    )

    approval_agent = Agent(
        role="Approval Agent",
        goal="Apply scoring rules on the calculator output, normalize the score between 0 and 10, "
             "and return loan approval status with reasons.",
        backstory=f"You are the loan manager who uses the generated metrics to decide whether "
                  f"customer {customer_id} should be approved.",
        llm = ollm
    )
    
    return input_agent, calculator_agent, approval_agent

def create_task(customer_id):
    input_agent, calculator_agent, approval_agent = create_agents(customer_id)
    input_agent_task = Task(
        description=f"Fetch bank statements, credit card details, and loan data for customer {customer_id}.",
        expected_output="A JSON object containing all financial data for the customer.",
        agent=input_agent
    )

    calculator_agent_task = Task(
        description="""Process the input data and compute metrics strictly following the exact formula
                        ## 1. Income & Spend Metrics
•⁠  ⁠Count all transactions:
    - credit → contributes to monthly income
    - debit → contributes to monthly spend
•⁠  ⁠Calculate:
    - avg_monthly_income = total_credit_amount / number_of_credit_transactions
    - avg_monthly_spend = total_debit_amount / number_of_debit_transactions
    - total_transaction_amount = income + spend
    - avg_transaction_amount = total_transaction_amount / total_transactions
    - spend_ratio = (avg_monthly_spend / avg_monthly_income)  100

---

## 2. Spending Variance & Anomaly Detection
•⁠  ⁠Compute variance of transaction amounts:
    - variance = average of (avg_transaction_amount – transaction_amount)²
    - standard_deviation = sqrt(variance)
•⁠  ⁠anomaly_flags = number of transactions where amount > standard_deviation

---

## 3. Credit Card Late Payments (from credits_loan.json)
For each customer:
•⁠  ⁠Count ⁠late_payment_count by checking each billing cycle:
    - If payment_date > cycle_end OR amount_paid < amount_due → late payment

---

## 4. Current Active Loans
•⁠  ⁠current_loans = number of loans where outstanding_amount > 0

---

## 5. Credit Utilization Ratio
For each credit card:
    utilization_percent = (current_balance / credit_limit) × 100
•⁠  ⁠credit_utilization_ratio = average of all utilization values

---

## 6. Account Age
•⁠  ⁠account_age_months = difference between today and account_creation_date in months

---


        """,
        expected_output= """Return exact json 
         {
            "metrics": {
            "avg_month_income": 0.0,
            "avg_month_spend": 0.0,
            "late_payment_count": 0,
            "credit_utilization_ratio": 0.0,
            "account_age_months": 0,
            "anomaly_flags": 0,
            "transaction_pattern": {
                "incoming_to_outgoing_ratio": 0.0,
                "anomaly_flags": 0
            }
            },
            "metrics_notes": [
            "optional list of parsing/assumption notes"
            ]
            } """,
        agent=calculator_agent,
        context = [input_agent_task]
    )

    approval_agent_task = Task(
        description="""Take the computed parameters and apply these decision rules to Provide approval or rejection with explanation
        
        
                REJECT if ANY:
        - late_payment_count > 3
        - credit_utilization_ratio > 70
        - anomaly_flags > 0
        - avg_month_income < avg_month_spend
        - account_age_months < 3
        - transaction_pattern.anomaly_flags > 0

        APPROVE if ALL:
        - income_spend_ratio >= 1.5
        - late_payment_count <= 1
        - credit_utilization_ratio <= 40
        - account_age_months >= 12
        - anomaly_flags == 0
        - transaction_pattern.incoming_to_outgoing_ratio >= 1.2  (treat inf as > 1.2)

        REVIEW if ALL:
        - 1.1 <= income_spend_ratio < 1.5
        - late_payment_count <= 2
        - credit_utilization_ratio <= 60
        - anomaly_flags == 0

        
        
        """,
        expected_output= """ Return EXACT JSON:
{
  "customer_id": "...",
  "decision": "APPROVE" | "REJECT" | "REVIEW",
  "reason": "human readable explanation for the decision (include failing checks)",
  "income_spend_ratio": number,
  "metrics": {
    "avg_month_income": number,
    "avg_month_spend": number,
    "late_payment_count": integer,
    "credit_utilization_ratio": number,
    "account_age_months": integer,
    "anomaly_flags": integer,
    "transaction_pattern": {
      "incoming_to_outgoing_ratio": number,
      "anomaly_flags": integer
    }
  },""" ,
        agent=approval_agent,
        context = [calculator_agent_task]
    )

    return input_agent_task, calculator_agent_task, approval_agent_task



customer_id = input("Enter the customer id: ")
input_agent, calculator_agent, approval_agent = create_agents(customer_id)
input_agent_task, calculator_agent_task, approval_agent_task = create_task(customer_id)


crew = Crew(
    agents=[input_agent, calculator_agent, approval_agent],
    tasks=[input_agent_task, calculator_agent_task, approval_agent_task],
    verbose=True
)

crew.kickoff()