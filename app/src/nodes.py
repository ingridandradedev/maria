import os
import logging
import requests
import json
from typing import Dict, Any
from datetime import datetime

# Bibliotecas para PDF
import pdfkit
import jinja2

# Google Cloud
from google.cloud import storage
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from google.cloud import speech_v1p1beta1 as speech

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Configurações do Google Cloud Storage
BUCKET_NAME = "maria-1-0-pecege"
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "maria-456618-871b8f622168.json")

def transcribe_audio(state: Dict[str, Any]):
    """
    Transcreve um arquivo de áudio usando Google Speech-to-Text v1 com diarization.
    Suporta URLs assinadas ao baixar o arquivo localmente.
    """
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code="pt-BR",
        enable_automatic_punctuation=True,
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=2,
            max_speaker_count=5  # ajuste conforme necessário
        )
    )

    try:
        audio_file_url = state["audio_file"]
        logging.debug(f"Baixando arquivo de áudio da URL: {audio_file_url}")

        # Baixa o arquivo de áudio localmente
        local_audio_path = "temp_audio.mp3"
        with requests.get(audio_file_url, stream=True) as response:
            response.raise_for_status()
            with open(local_audio_path, "wb") as audio_file:
                for chunk in response.iter_content(chunk_size=8192):
                    audio_file.write(chunk)

        logging.debug(f"Arquivo de áudio baixado com sucesso: {local_audio_path}")

        # Lê o conteúdo do arquivo de áudio
        with open(local_audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        # Configura o áudio para o Google Speech-to-Text
        audio = speech.RecognitionAudio(content=audio_content)

        logging.debug("Iniciando transcrição de áudio com diarization...")

        # Envia a solicitação para o Google Speech-to-Text
        operation = client.long_running_recognize(
            request={
                "config": config,
                "audio": audio
            }
        )

        logging.debug("Aguardando conclusão da transcrição...")
        response = operation.result(timeout=300)

        full_transcript_with_speakers = []

        for result in response.results:
            alternative = result.alternatives[0]

            if alternative.words:
                current_speaker = alternative.words[0].speaker_tag
                current_transcript = []

                for word_info in alternative.words:
                    speaker_tag = word_info.speaker_tag
                    word = word_info.word

                    if speaker_tag != current_speaker:
                        # salva o trecho anterior
                        full_transcript_with_speakers.append(
                            f"Falante {current_speaker}: {' '.join(current_transcript)}"
                        )
                        current_transcript = []
                        current_speaker = speaker_tag

                    current_transcript.append(word)

                # salva o último trecho
                if current_transcript:
                    full_transcript_with_speakers.append(
                        f"Falante {current_speaker}: {' '.join(current_transcript)}"
                    )
            else:
                full_transcript_with_speakers.append(alternative.transcript)

        full_transcript = "\n".join(full_transcript_with_speakers)

        logging.debug(f"Transcrição com diarization concluída. Tamanho: {len(full_transcript)} caracteres")

        # Remove o arquivo local após a transcrição
        os.remove(local_audio_path)

        return {
            "meeting_transcript": full_transcript,
            "messages": [
                {"role": "ai", "content": f"Transcrição com diarization concluída: {full_transcript[:200]}..."}
            ]
        }

    except Exception as e:
        logging.error(f"Erro detalhado na transcrição: {e}", exc_info=True)
        return {
            "meeting_transcript": "",
            "messages": [
                {"role": "ai", "content": f"Erro na transcrição: {str(e)}"}
            ]
        }

def load_gemini_model():
    from vertexai import init
    init(
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

def generate_meeting_summary(state: Dict[str, Any]):
    """
    Gera o resumo da transcrição de reunião usando Gemini
    """
    model = load_gemini_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente especializado em resumir transcrições de reuniões."),
        ("human", "Resuma a seguinte transcrição de reunião:\n\n{transcript}")
    ])

    chain = prompt | model | StrOutputParser()

    summary = chain.invoke({"transcript": state["meeting_transcript"]})

    return {
        "meeting_summary": summary,
        "messages": [
            *state.get("messages", []),
            {"role": "ai", "content": summary}
        ]
    }

def generate_feedback(state: Dict[str, Any]):
    """
    Gera o feedback estruturado usando Gemini
    """
    model = load_gemini_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Aja como um(a) gestor(a) experiente responsável por conduzir uma conversa de feedback estruturado com um(a) colaborador(a). 
        Com base nas informações da transcrição, preencha os campos da ficha de feedback de forma objetiva, respeitosa e construtiva."""),
        ("human", """Transcrição da reunião:
        {transcript}

        Preencha os seguintes campos em formato JSON:
        - data: Data atual
        - nome_do_liderado: Nome do colaborador
        - nome_do_lider: Nome do líder
        - passo1: Reconhecimento e valorização do potencial
        - passo2: Descrição do comportamento atual
        - passo3: Cenário esperado após mudança
        - passo4: Reflexão sobre causas
        - passo5: Ações de mudança propostas pelo liderado
        - passo6: Ações sugeridas pelo líder
        - pontos_fortes: Lista de pontos fortes
        - exemplo_pontos_fortes: Exemplos dos pontos fortes
        - pontos_fracos: Lista de pontos fracos
        - exemplo_pontos_fracos: Exemplos dos pontos fracos
        """)
    ])

    chain = prompt | model | StrOutputParser()

    try:
        feedback_json_str = chain.invoke({
            "transcript": state["meeting_transcript"]
        })
        
        # Tenta parsear o JSON
        feedback_content = json.loads(feedback_json_str)
        
        return {
            "feedback_content": feedback_content,
            "messages": [
                *state.get("messages", []),
                {"role": "ai", "content": "Feedback gerado com sucesso"}
            ]
        }
    except Exception as e:
        logging.error(f"Erro ao gerar feedback: {e}")
        return {
            "feedback_content": {},
            "messages": [
                *state.get("messages", []),
                {"role": "ai", "content": f"Erro ao gerar feedback: {str(e)}"}
            ]
        }

def generate_pdf_and_upload(state: Dict[str, Any]):
    """
    Gera um PDF com o feedback estruturado e faz upload para o Google Cloud Storage
    """
    feedback = state["feedback_content"]

    # Carrega o template HTML
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("feedback_template.html")

    # Renderiza o HTML com os dados do feedback
    html_content = template.render(feedback)

    # Gera o PDF
    pdf_filename = "feedback.pdf"
    pdfkit.from_string(html_content, pdf_filename)

    # Faz upload para o Google Cloud Storage
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(pdf_filename)

    blob.upload_from_filename(pdf_filename)

    # Gera uma URL assinada válida por 1 hora
    url_assinada = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="GET"
    )

    # Remove o arquivo local
    os.remove(pdf_filename)

    return {
        "pdf_url": url_assinada,
        "messages": [
            *state.get("messages", []),
            {"role": "ai", "content": "PDF gerado e enviado com sucesso"}
        ]
    }