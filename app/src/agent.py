from langgraph.graph import StateGraph, START, END
from src.state import MeetingSummaryState
from src.nodes import transcribe_audio, generate_feedback, generate_pdf_and_upload

def create_meeting_summary_graph():
    """
    Cria o grafo de fluxo para feedback estruturado
    """
    graph_builder = StateGraph(MeetingSummaryState)

    # Adiciona os nós
    graph_builder.add_node("transcribe_audio", transcribe_audio)
    graph_builder.add_node("generate_feedback", generate_feedback)
    graph_builder.add_node("generate_pdf_and_upload", generate_pdf_and_upload)

    # Define as arestas
    graph_builder.add_edge(START, "transcribe_audio")
    graph_builder.add_edge("transcribe_audio", "generate_feedback")
    graph_builder.add_edge("generate_feedback", "generate_pdf_and_upload")
    graph_builder.add_edge("generate_pdf_and_upload", END)

    # Compila o grafo
    return graph_builder.compile()

# Cria a instância do grafo
graph = create_meeting_summary_graph()