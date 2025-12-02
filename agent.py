import json
import os
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)

def create_agents():
    data_agent = Agent(
        role="Financial Data Processor",
        goal="Fetch customer data and compute all credit metrics.",
        backstory=(
            "You load bank statements, loans, and credit cards, and transform raw "
            "data into meaningful credit metrics used for scoring."
        ),
        verbose = True,
        llm=llm
    )

    scoring_agent = Agent(
        role="Credit Decision Engine",
        goal="Produce a final credit decision using the metrics.",
        backstory="A risk analyst who turns metrics into a credit score and approval decision.",
        llm=llm
    )

    return data_agent, scoring_agent

def create_tasks(customer_id):
    data_agent, scoring_agent = create_agents()

    data_task = Task(
        description=f"""
        You must do 2 things:

        ------------------------------
        1) FETCH CUSTOMER DATA
        ------------------------------
        • Load bank_statements.json
        • Load credits_loan.json
        • Match using {customer_id}
        • Combine into a single JSON:
            "customer_id": "...",
            "transactions": [...],
            "credit_cards": [...],
            "loans": [...],
            "account_creation_date": "YYYY-MM-DD"

        ------------------------------
        2) CALCULATE ALL METRICS
        ------------------------------
        Compute:
        - avg_month_income
        - avg_month_spend
        - transaction_count
        - avg_transaction_amount
        - spend_ratio
        - anomaly_flags (values far from mean)
        - credit_utilization_ratio
        - late_payment_count
        - active_loans
        - account_age_months

        Output ONLY a clean JSON:
        "customer_id": "...",
        "avg_month_income": ...,
        "avg_month_spend": ...,
        "transaction_count": ...,
        "avg_transaction_amount": ...,
        "spend_ratio": ...,
        "anomaly_flags": ...,
        "credit_utilization_ratio": ...,
        "late_payment_count": ...,
        "active_loans": ...,
        "account_age_months": ...
        """,
        expected_output="A JSON object with all computed metrics",
        verbose = True,
        agent=data_agent
    )

    scoring_task = Task(
        description=f"""
        You will receive a metrics JSON with fields:
        - avg_month_income
        - avg_month_spend
        - late_payment_count
        - credit_utilization_ratio
        - account_age_months
        - anomaly_flags
        - transaction_pattern {{"incoming_to_outgoing_ratio": ..., "anomaly_flags": ...}}

        Apply the EXACT decision rules:

        -------------------------------------------------------
        Compute:
        income_spend_ratio = income / spend (if spend > 0 else 0)
        -------------------------------------------------------

        REJECT if ANY of these are true:
        - late > 3
        - util > 70
        - anomaly_flags > 0
        - income < spend
        - account_age_months < 3
        - transaction_pattern.anomaly_flags > 0

        APPROVE if ALL are true:
        - income_spend_ratio >= 1.5
        - late <= 1
        - util <= 40
        - account_age_months >= 12
        - anomaly_flags == 0
        - transaction_pattern.incoming_to_outgoing_ratio >= 1.2

        REVIEW if ALL are true:
        - 1.1 <= income_spend_ratio < 1.5
        - late <= 2
        - util <= 60
        - anomaly_flags == 0

        If none of the above rules matched → decision = REVIEW.

        Return JSON:
        {{
        "customer_id": "...",
        "decision": "APPROVE / REJECT / REVIEW",
        "reason": "...",
        "income_spend_ratio": ...
        }}
        """,
        expected_output="Final credit decision JSON",
        agent=scoring_agent
    )

    return data_task, scoring_task

customer_id = input("Enter customer ID: ")

data_agent, scoring_agent = create_agents()
data_task, scoring_task = create_tasks(customer_id)


crew = Crew(
    agents=[data_agent, scoring_agent],
    tasks=[data_task, scoring_task],
    verbose=True
)

result = crew.kickoff()

print("\n🎯 FINAL CREDIT DECISION:\n", result)
