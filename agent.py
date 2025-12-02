import json
import os
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from crewai.tools import BaseTool

load_dotenv()

DATA_FETCHER_PROMPT = r"""
You are the Data Fetcher agent. Your job is to locate and return the exact account data for the requested `customer_id` from two JSON sources: `bank_statement.json` and `credits_loan.json`.

Requirements (strict):
1. Extract `customer_id` from the natural-language input. Accept formats like:
   - `customer 12345`
   - `customer_id: c-001`
   - `id=12345`
   If you cannot find an ID, respond with an error string: "error: customer_id not found in input".

2. From bank_statement.json find the entry whose `customer_id` (or `id`) exactly matches the requested id. Return its `transactions` field (or try txns/statement/entries). If still missing, return empty list and set transactions_present=false.

3. From credits_loan.json find the matching entry and extract:
   - account_creation_date (try account_creation_date, created_at, created)
   - credit_cards (try credit_cards, cards, credit_card)
   - loans
   Missing fields -> null or empty list accordingly.

4. Return JSON under key "fetched_customer_data" with exact shape:
{
  "customer_id": "string",
  "transactions": [ ... ],
  "account_creation_date": "YYYY-MM-DD or null",
  "credit_cards": [ ... ],
  "loans": [ ... ],
  "meta": {
    "transactions_present": true|false,
    "credit_cards_present": true|false,
    "loans_present": true|false,
    "notes": "explain any missing fields or parsing decisions"
  }
}

5. DO NOT compute metrics. Only fetch, normalize and return the data. Use ISO date strings where possible.
"""

SCORING_PROMPT = r"""
You are the Scoring Agent. You will receive `fetched_customer_data` JSON produced by the Data Fetcher. Compute metrics using the exact formulas and apply the exact decision rules below. Return ONLY the final decision JSON (no commentary).

Parsing & formulas:
- Determine transaction amount from amount/value/txn_amount/amt or credit/debit fields. If not parseable, treat as 0 and add note.
- Determine direction from type/txn_type/direction (credit/incoming => incoming; debit/outgoing => outgoing). If ambiguous, use sign (positive => incoming, negative => outgoing), else treat as outgoing.
- Group transactions by calendar month of date (or current month if date missing).
Formulas:
  avg_month_income = sum(incoming_amounts) / months_count (months_count = number of months with any transactions or 1)
  avg_month_spend  = sum(outgoing_amounts) / months_count (same months_count)
  late_payment_count = count(transactions where status contains late/overdue/missed or 'late' field truthy)
  For credit cards:
    total_limit = sum(card.limit)
    total_outstanding = sum(card.outstanding)
    credit_utilization_ratio = (total_outstanding / total_limit) * 100 (if total_limit == 0 -> set to 0.0 and add metrics_note)
  account_age_months = whole months between account_creation_date and today (0 if missing; note if missing)
  anomaly_flags = explicit flagged/anomaly/suspicious count + count(outgoing > 3 * median(outgoing_amounts)) (skip median step if no outgoings)
  transaction_pattern.incoming_to_outgoing_ratio = total_incoming / total_outgoing (if total_outgoing==0 and total_incoming>0 -> inf; if both 0 -> 0.0)
  income_spend_ratio = avg_month_income / avg_month_spend (if avg_month_spend==0 and avg_month_income>0 -> inf; if both 0 -> 0)

Decision rules (APPLY EXACTLY):
Compute income_spend_ratio as above.

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

If none matches exactly -> decision = REVIEW.

Return EXACT JSON:
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
  },
  "metrics_notes": [ ... ]
}
"""


class fetch_tool(BaseTool):
    name:str = "Data fetcher"
    description:str = "Fetches the customer account data, from database(json file)"
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

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 
llm = LLM(
    model="ollama/phi3",
    base_url="http://localhost:11434"

)

def create_agents():
    input_agent = Agent(
        role="Financial Data Processor",
        goal="Fetch customer data and return normalized fetched_customer_data JSON",
        backstory="Expert in extracting exact records from JSON files",
        tools=[fetch_tool()],
        verbose=True,
        allow_delegation = True,
        llm=llm
    )

    scoring_agent = Agent(
        role="Credit Decision Engine",
        goal="Compute metrics from fetched_customer_data and return final decision JSON",
        backstory="Risk analyst that applies deterministic rules to metrics",
        verbose = True,
        llm=llm
    )

    return input_agent, scoring_agent

def create_tasks(natural_input):
    input_agent, scoring_agent = create_agents()

    input_task = Task(
        description=DATA_FETCHER_PROMPT + "\n\nNaturalInput:\n" + natural_input,
        expected_output="A json object with key 'fetched_customer_data' containing normalized data",
        agent=input_agent
    )

    scoring_task = Task(
        description=SCORING_PROMPT + "\n\nInput will be the fetched_customer_data JSON returned by the Data Fetcher agent.",
        expected_output="Final credit decision JSON",
        agent=scoring_agent,
        context=[input_task]
    )

    return input_task, scoring_task

# ---------------- Orchestration / kickoff ----------------

natural_input = input("Enter the requirements (e.g. 'Fetch customer C101'): ")

input_agent, scoring_agent = create_agents()
input_task, scoring_task = create_tasks(natural_input)

crew = Crew(
    agents=[input_agent, scoring_agent],
    tasks=[input_task, scoring_task],
    verbose=True
)

result = crew.kickoff()

print("\n FINAL RESULT FROM CREW:")
try:
    print(json.dumps(result, indent=2, default=str))
except Exception:
    print(result)
