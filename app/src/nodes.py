import os
import logging
import requests
import json
from typing import Dict, Any
from datetime import datetime, timedelta

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

# Configurações do Google Cloud Storage
BUCKET_NAME = "maria-1-0-pecege"

# Substitua o conteúdo abaixo pelo JSON das credenciais
CREDENTIALS_JSON = {
    "type": "service_account",
    "project_id": "maria-456618",
    "private_key_id": "871b8f622168f17cd1b863d59f46657e977ee091",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCdXiAglMf4XG3f\ndGMJY4V/U+1AMcalR7MKjBANuKZa+D12x3Y/fTcbnivyO7sm1hp3WIZyDfNR6f7o\nJuh/Y/HkVeCDY4IwdHkZkUvcVojBahq6IeefEWi6n1kx4dmdGVBtZo1sQHT4t+wr\n2SBMj4JC4OcDzj8fgNXqJ3kISLK+wuMZqD6PtQtyKgp9rD9y9bGhccOeIEmmHdPt\ntYwdMCedJBdRfoMA+OK2HMBmdvhOWwdolmPJY6mxJMoAriS8FYXy1SGWis3eOSwB\n/hA3Gey2OxbLWE5nt3X9gmN/k9lZM796VUEycJj+cjaBdCnHVUp+YCe7NmPzf2pj\nkexl4VmLAgMBAAECggEAARCBhpq0LfYFVawRtKsl1lYO7TA2uMze9IckcbR1pPKZ\no3Rn5sQvIzidwKzJ81JHGnU7VtKYVcfSu/NTbQwHw3GdvmH4PlttPc2Uu2Fq8ewt\nUOpKgt+6oVF+ALPgkETix/vO6eZsq8BpB1kXt0qTmceXoZgHxeXBtEsuG6u8sft9\nBZ9T/SHRQJCpHfuxQaHKpAmW5lwufK92sIQIkc+D+qgaIPRf6vxoJUeRDqaBlTKb\nK3uGjIkQr+QKA6xcbgL9cBsXPp4cufv4z8XKNMAGQVCJTRA/ybNgw8uOoZVjA63b\nPyQxYK0fwKcfdS6CyrsUqdiyHzwxYyoGQFsU1lfXJQKBgQDJxTf5QMUAvQphmjc7\nI7nmkpudhCJSSFGjzFd2ovNyYnjv0b3Q52gS77VRnHtFfuHng6zZfg6J4uoztKxY\nGsdSuiZwZHVoE25gpxhrjPJ34xB5WSD8RDuOG6E3yTRcIkmuFw8rDhrjh88Czb27\n6YGYxwESq2gIFz35938tWZGUTwKBgQDHqckACM3Ny/v4l9zoC8BYR84IGhwqbQvy\ni4IAx0flyfIE1OMS9lvCW6QiZVCTiV2pOhnExscE/gBKGU1uJiS7x7Nl+eBRl/TA\na6ruju9Je8WgsDlxDHfNBqpvihbumGA7WipUSELRKOpVHat47sVfl4pUHXhJHrdV\ngFOPN15MBQKBgAPmly1vbh+UiAXZCGZRS3/Ep9OEwXEbytBC3BIEFnbIppPkVyoZ\nvy5WigfY4Z03VcC3D/locXmC9IopXQebBO15gdK8bnSjo4ek01kI8YsVzbS632Nh\nIlGeASDl9+gsFYaTFYz8idKKRptERP3EBuhgOIoW3D0DzgPuH/xNdf4LAoGBAKuK\nUei3p6nyYW95eg/bWMwAFSGc3SoOOj+OYIkurbTdRhOkm9tE0h0wAtqSVSIM2O11\nv4HyjjbZy4HeL0o9dz3mG1m3z2QKu/s+BcOkBi4KKwcdoJxh6+O4oGHoMD+ZpsQX\neVqkItP05S9vqEzkR6sTVYNjNl8MBtNsx126YkfJAoGAEWOWscaICTWOevtd7brN\n5JolHk4/WstMBf8HxRF1+n9aq4iOP3F/65ukbfLzDlpy+3hIaTZT/V4uDHZa0xRv\nlhK5GUgNz1qWBOgOR2INraHxS5szA1emGkPHkLEeuQLLhuz9oqE3TqisIldZQ0Jq\nI/Z/qHGdv2WZF/WdxXhcK6w=\n-----END PRIVATE KEY-----\n",
    "client_email": "maria-31@maria-456618.iam.gserviceaccount.com",
    "client_id": "103963023317629107070",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/maria-31%40maria-456618.iam.gserviceaccount.com"
}

# Carrega as credenciais diretamente do JSON
credentials = service_account.Credentials.from_service_account_info(CREDENTIALS_JSON)

# Carrega variáveis de ambiente
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

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
    Gera um PDF com o feedback estruturado e faz upload para o Google Cloud Storage
    """
    feedback = state["feedback_content"]
    template_path = os.path.join(os.path.dirname(__file__), "templates", "feedback_template.html")
    logging.debug(f"Caminho do template HTML: {template_path}")

    # Carrega o template HTML
    with open(template_path, "r", encoding="utf-8") as file:
        template_content = file.read()

    # Renderiza o template com os dados do feedback
    template = jinja2.Template(template_content)
    rendered_html = template.render(feedback)

    # Caminho absoluto do executável wkhtmltopdf
    config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")
    options = {
        "enable-local-file-access": ""  # Permite acesso a arquivos locais
    }

    # Gera o PDF
    pdf_path = "feedback.pdf"
    pdfkit.from_string(rendered_html, pdf_path, configuration=config, options=options)

    # Após gerar o PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"O arquivo PDF {pdf_path} não foi encontrado.")
    else:
        logging.debug(f"PDF gerado com sucesso: {pdf_path}")

    # Faz upload para o Google Cloud Storage
    try:
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(pdf_path)

        # Antes do upload
        logging.debug(f"Fazendo upload do PDF para o bucket {BUCKET_NAME}...")
        blob.upload_from_filename(pdf_path)
        logging.debug("Upload concluído com sucesso.")

        # Gera uma URL assinada válida por 1 hora
        url_assinada = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )
        logging.debug(f"URL assinada gerada com sucesso: {url_assinada}")

        # Remove o arquivo local
        os.remove(pdf_path)

        return {
            "pdf_url": url_assinada,
            "messages": [
                *state.get("messages", []),
                {"role": "ai", "content": "PDF gerado e enviado com sucesso"}
            ]
        }

    except Exception as e:
        logging.error(f"Erro ao fazer upload ou gerar URL assinada: {e}", exc_info=True)
        raise