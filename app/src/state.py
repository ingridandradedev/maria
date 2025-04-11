from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import Message  # Ajuste o caminho conforme necessário
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import Message
# OU
from langchain_core.messages import BaseMessage


class MeetingSummaryState(TypedDict):
    """
    Estado para gerenciar o fluxo de resumo de reunião
    """
    messages: Annotated[List[BaseMessage], add_messages]
    meeting_transcript: str
    meeting_summary: str