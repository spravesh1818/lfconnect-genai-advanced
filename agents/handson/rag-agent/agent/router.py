from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def route_question(state):
    prompt = f"""
    Classify the question:

    {state['question']}

    Respond ONLY with:
    - retrieve
    - general

    Example:
    Question: What is the capital of France?
    general

    Question: What is the 13th amendment to the Nepal Constitution?
    retrieve
    """

    decision = llm.invoke(prompt).content.strip().lower()
    return {"route": decision}
