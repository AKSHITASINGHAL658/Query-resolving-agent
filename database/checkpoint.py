# database/checkpoint.py
from contextlib import ExitStack
from langgraph.checkpoint.sqlite import SqliteSaver
 
# Task 7 / Section 10.5: Database storage file for persistent session tracking
DB_PATH = "customer_support_memory.db"
 
_stack = ExitStack()
 
# This memory object is what gets imported and passed to builder.compile()
memory = _stack.enter_context(
    SqliteSaver.from_conn_string(DB_PATH)
)