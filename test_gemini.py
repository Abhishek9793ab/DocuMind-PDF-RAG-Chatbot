import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
for model in [os.getenv("GEMINI_MODEL", "gemini-3.6-flash"), "gemini-3.6-flash", "gemini-3.5-flash"]:
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"), max_retries=1, timeout=30)
        print(model, "->", llm.invoke("Say hello in one word").content)
        break
    except Exception as e:
        print(model, "FAILED:", str(e)[:150])