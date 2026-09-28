import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests

# LangChain / LangGraph Imports
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

# 1. Setup & Environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "C9PE94QUE9VWGFM")

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# 2. Define Tools
@tool
def get_stock_price(symbol: str) -> str:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        r = requests.get(url)
        data = r.json().get("Global Quote", {})
        if not data: return f"Could not find price data for {symbol}."
        price = data.get("05. price", "N/A")
        return f"The current stock price of {symbol} is ${price}."
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """Simulates purchasing a stock. Requires human approval."""
    decision = interrupt(f"CONFIRMATION REQUIRED: Do you approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return f"SUCCESS: Purchase order placed for {quantity} shares of {symbol}."
    return f"CANCELLED: Purchase of {quantity} shares of {symbol} was declined."

# 3. Graph Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_chatbot():
    """Creates and returns the compiled LangGraph chatbot."""
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    tools = [get_stock_price, purchase_stock]
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: ChatState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(ChatState)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "chat_node")
    workflow.add_conditional_edges("chat_node", tools_condition)
    workflow.add_edge("tools", "chat_node")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Create a global instance for import into streamlit_app.py
chatbot = create_chatbot()

# 4. Execution Loop (for terminal use)
def run_chatbot():
    thread_id = "user-session-1"
    config = {"configurable": {"thread_id": thread_id}}
    print("--- Stock Bot Started (Type 'exit' to quit) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        snapshot = chatbot.get_state(config)
        if snapshot.next:
            interrupt_val = None
            if snapshot.tasks:
                for task in snapshot.tasks:
                    if task.interrupts:
                        interrupt_val = task.interrupts[0].value
                        break
            if interrupt_val:
                print(f"\n🚨 HITL: {interrupt_val}")
                decision = input("Your decision (yes/no): ").strip().lower()
                result = chatbot.invoke(Command(resume=decision), config=config)
        last_msg = result["messages"][-1]
        print(f"Bot: {last_msg.content}\n")

if __name__ == "__main__":
    run_chatbot()
