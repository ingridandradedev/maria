#agent.py
from langgraph.graph import StateGraph, START, END
from src.state import MeetingSummaryState
from src.nodes import generate_meeting_summary

def create_meeting_summary_graph():
    """
    Cria o grafo de fluxo para resumo de reunião
    """
    graph_builder = StateGraph(MeetingSummaryState)

    # Adiciona o nó de geração de resumo
    graph_builder.add_node("generate_summary", generate_meeting_summary)

    # Define o ponto de entrada
    graph_builder.add_edge(START, "generate_summary")

    # Define o ponto de saída
    graph_builder.add_edge("generate_summary", END)

    # Compila o grafo
    return graph_builder.compile()

# Cria a instância do grafo
graph = create_meeting_summary_graph()
