#nodes.py
import os
import vertexai
from typing import Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

def load_gemini_model():
    """Carrega o modelo Gemini 1.5 Pro"""
    # Inicializa o Vertex AI com o project ID
    vertexai.init(
        project=os.getenv('GOOGLE_CLOUD_PROJECT'), 
        location="us-central1"
    )
    
    return ChatVertexAI(
        model_name="gemini-1.5-pro",
        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
        location="us-central1",
        temperature=0.3,
        max_tokens=2048
    )

def prepare_summary_prompt() -> ChatPromptTemplate:
    """Cria o prompt para resumo de reunião"""
    return ChatPromptTemplate.from_messages([
        ("system", """Você é um agente de inteligência artificial especializado em resumir reuniões corporativas. 
        Sua função é transformar transcrições de reuniões em um resumo executivo claro, organizado e útil para consulta posterior.

        Com base na transcrição recebida, gere um resumo estruturado da reunião contendo os seguintes tópicos:

        1. Objetivo da Reunião
        Descreva de forma objetiva o motivo principal da reunião.

        2. Pauta (Assuntos abordados)
        Liste os tópicos principais discutidos.

        3. Resumo das Discussões
        Para cada item da pauta, descreva os principais pontos debatidos.

        4. Decisões Tomadas
        Aponte acordos, definições e resoluções ocorridas durante a reunião.

        5. Encaminhamentos
        Liste ações definidas, quem será o responsável e prazos (se mencionados).

        6. Próximos Passos (se houver)
        Aponte o que deve ser feito a seguir ou o que será discutido na próxima reunião.
        """),
        ("human", "Transcrição da reunião:\n{meeting_transcript}")
    ])

def generate_meeting_summary(state: Dict[str, Any]):
    """
    Gera o resumo da reunião usando o modelo Gemini
    """
    model = load_gemini_model()
    prompt = prepare_summary_prompt()
    
    # Cria o chain de processamento
    chain = prompt | model | StrOutputParser()
    
    # Gera o resumo
    summary = chain.invoke({
        "meeting_transcript": state['meeting_transcript']
    })
    
    return {
        "meeting_summary": summary,
        "messages": [
            {"role": "ai", "content": summary}
        ]
    }