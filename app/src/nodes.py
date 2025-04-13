import os
import logging
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
    Suporta arquivos do Google Cloud Storage.
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
        logging.debug(f"Iniciando transcrição de áudio com diarization: {state['audio_file']}")

        operation = client.long_running_recognize(
            request={
                "config": config,
                "audio": speech.RecognitionAudio(uri=state["audio_file"])
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
