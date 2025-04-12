# agent.py
from langgraph.graph import StateGraph, START, END
from src.state import MeetingSummaryState
from src.nodes import transcribe_audio, generate_meeting_summary

def create_meeting_summary_graph():
    """
    Cria o grafo de fluxo para resumo de reunião
    """
    graph_builder = StateGraph(MeetingSummaryState)

    # Adiciona os nós
    graph_builder.add_node("transcribe_audio", transcribe_audio)
    graph_builder.add_node("generate_summary", generate_meeting_summary)

    # Define as arestas
    graph_builder.add_edge(START, "transcribe_audio")
    graph_builder.add_edge("transcribe_audio", "generate_summary")
    graph_builder.add_edge("generate_summary", END)

    # Compila o grafo
    return graph_builder.compile()

# Cria a instância do grafo
graph = create_meeting_summary_graph()
