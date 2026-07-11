from langchain_ollama import ChatOllama
from rich.console import Console
console = Console()
llm=ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3
)