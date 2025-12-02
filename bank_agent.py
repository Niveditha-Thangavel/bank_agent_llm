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
        goal=f"Gather all financial data for customer {customer_id} from the database: "
             "bank statements, credit card details, and loan details.",
        backstory=f"You are an information fetcher that accurately retrieves all the "
                  f"financial records for customer {customer_id}.",
        tools = [fetch_tool()],
        allow_delegation = True,
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
        description="Process the input data and compute metrics such as average monthly income, "
                    "average spend, transaction patterns, and credit utilization ratio.",
        expected_output="A structured JSON object containing processed credit-scoring parameters.",
        agent=calculator_agent,
        context = [input_agent_task]
    )

    approval_agent_task = Task(
        description="Take the computed parameters and calculate credit score (0–10). "
                    "Provide approval or rejection with explanation.",
        expected_output="Loan approval decision, score, and reason.",
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