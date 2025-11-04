pip install langchain langchain-text-splitters langchain-community bs4
pip install -U langchain-google-genai

import os
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
from langchain.schema import HumanMessage

const model = new ChatGoogleGenerativeAI({
modelName: "gemini-2.5-flash-lite",
temperature: 0
});