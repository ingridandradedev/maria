from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MeetingSummaryState(TypedDict):
    """
    Estado para gerenciar o fluxo de feedback estruturado
    """
    messages: Annotated[List[BaseMessage], add_messages]
    audio_file: str  # Arquivo de áudio
    meeting_transcript: str  # Transcrição da reunião
    feedback_content: dict  # Feedback estruturado gerado
    pdf_url: str  # URL do PDF gerado