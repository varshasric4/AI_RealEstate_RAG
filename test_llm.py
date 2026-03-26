from config import *
from langchain_openai import ChatOpenAI

print("Testing LLM connection...")
print(f"Model: {MODEL_NAME}")
print(f"Base URL: {MODEL_BASE_URL}")
print(f"API Key starts with: {OPENROUTER_API_KEY[:10]}...")

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_base=MODEL_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.3
)

try:
    response = llm.invoke("Say hello in one word")
    print(f"✅ LLM works! Response: {response.content}")
except Exception as e:
    print(f"❌ LLM FAILED!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")