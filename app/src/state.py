#state.py
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class MeetingSummaryState(TypedDict):
    """
    Estado para gerenciar o fluxo de resumo de reunião
    """
    messages: Annotated[List[BaseMessage], add_messages]
    meeting_transcript: str
    meeting_summary: str