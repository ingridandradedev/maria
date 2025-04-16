import os
import logging
import requests  # Adicionado para baixar o arquivo
from typing import Dict, Any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.cloud import speech_v1p1beta1 as speech  # <- versão beta para diarization
from dotenv import load_dotenv

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