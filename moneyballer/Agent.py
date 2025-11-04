pip install langchain langchain-text-splitters langchain-community bs4
pip install -U "langchain-openai"


import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Assure-toi d'avoir exporté OPENAI_API_KEY dans ton environnement
# export OPENAI_API_KEY="sk-..."

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  # adapte model_name si besoin
resp = model([HumanMessage(content="Donne des information sur ce joueur de foot.")])
print(resp.content)
