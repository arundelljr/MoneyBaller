pip install langchain langchain-text-splitters langchain-community bs4
pip install -U langchain-google-genai

import os
from langchain_google_genai import ChatGoogleGenerativeAI

const model = new ChatGoogleGenerativeAI({
modelName: "gemini-2.5-flash-lite",
temperature: 0
});

resp = llm.invoke("Hello, world!")
print(resp.content)