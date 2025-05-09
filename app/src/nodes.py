import os
import logging
import requests
import json
from typing import Dict, Any
from datetime import datetime, timedelta

# PDF
import pdfkit
import jinja2

# GCP
from google.cloud import storage, secretmanager
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from google.cloud import speech_v1p1beta1 as speech

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# recupera o nome do secret (com versão) do .env
SECRET_NAME = os.getenv("GOOGLE_SECRET_NAME")
if not SECRET_NAME:
    raise RuntimeError("GOOGLE_SECRET_NAME não definido no .env")

def get_credentials() -> service_account.Credentials:
    client = secretmanager.SecretManagerServiceClient()
    # acessa a versão latest do secret
    response = client.access_secret_version(request={"name": SECRET_NAME})
    payload = response.payload.data.decode("utf-8")
    info = json.loads(payload)
    return service_account.Credentials.from_service_account_info(info)

# instancia única
credentials = get_credentials()

# bucket GCS
BUCKET_NAME = "projeto-maria-1-0-pecege"

logging.basicConfig(level=logging.DEBUG)

def transcribe_audio(state: Dict[str, Any]):
    """
    Transcreve um arquivo de áudio usando Google Speech-to-Text v1 com diarization.
    Suporta URLs assinadas ao baixar o arquivo localmente.
    """
    client = speech.SpeechClient(credentials=credentials)

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

    local_audio_path = None  # Inicializa a variável

    try:
        audio_source = state["audio_file"]

        # Se vier como gs://… passa direto para o Speech-to-Text
        if audio_source.startswith("gs://"):
            logging.debug(f"Usando URI GCS para áudio: {audio_source}")
            audio = speech.RecognitionAudio(uri=audio_source)

        else:
            # mantém o download e inline antigo para URLs assinadas ou HTTP(s)
            logging.debug(f"Baixando arquivo de áudio da URL: {audio_source}")
            local_audio_path = "temp_audio.mp3"
            with requests.get(audio_source, stream=True) as response:
                response.raise_for_status()
                with open(local_audio_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            with open(local_audio_path, "rb") as f:
                audio_content = f.read()
            os.remove(local_audio_path)
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
        # Aumenta timeout para suportar gravações longas (até 40+ minutos)
        response = operation.result(timeout=3600)

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

    finally:
        # Remove o arquivo local se ele foi criado
        if local_audio_path and os.path.exists(local_audio_path):
            os.remove(local_audio_path)

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

        Preencha os seguintes campos em formato JSON (NÃO IGNORE essa instrução: NÃO coloque ```json na sua resposta):
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
         
         NÃO IGNORE essa instrução: NÃO coloque ```json na sua resposta
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
                {"role": "ai", "content": f"Erro ao gerar feedback: {str(e)}"}
            ]
        }

def generate_pdf_and_upload(state: Dict[str, Any]):
    """
    Gera um PDF com o feedback estruturado e faz upload para o
    Google Cloud Storage, retornando a URL pública do objeto.
    """
    feedback = state["feedback_content"]
    template_path = os.path.join(
        os.path.dirname(__file__), "templates", "feedback_template.html"
    )
    logging.debug(f"Caminho do template HTML: {template_path}")

    # Renderiza o HTML
    with open(template_path, "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())
    rendered_html = template.render(feedback)

    # Configura e gera o PDF
    config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")
    pdf_path = "feedback.pdf"
    pdfkit.from_string(rendered_html, pdf_path, configuration=config, options={
        "enable-local-file-access": ""
    })

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"O arquivo PDF {pdf_path} não foi encontrado.")
    logging.debug(f"PDF gerado com sucesso: {pdf_path}")

    # Upload no GCS e obtém URL pública
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(pdf_path)

    logging.debug(f"Fazendo upload do PDF para o bucket {BUCKET_NAME}...")
    blob.upload_from_filename(pdf_path)
    logging.debug("Upload concluído com sucesso.")

    public_url = blob.public_url
    logging.debug(f"URL pública do PDF: {public_url}")

    # Remove o arquivo local
    os.remove(pdf_path)

    return {
        "pdf_url": public_url,
        "messages": [
            *state.get("messages", []),
            {"role": "ai", "content": "PDF gerado e enviado com sucesso"}
        ]
    }