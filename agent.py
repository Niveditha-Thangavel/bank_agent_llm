from crewai import Agent, Task, LLM

llm = LLM(
    model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)