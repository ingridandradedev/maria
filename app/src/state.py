from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MeetingSummaryState(TypedDict):
    """
    Estado para gerenciar o fluxo de resumo de reunião
    """
    messages: Annotated[List[BaseMessage], add_messages]
    audio_file: str  # Novo campo adicionado
    meeting_transcript: str
    meeting_summary: str
