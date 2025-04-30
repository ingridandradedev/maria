from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import graph

# Modelo de entrada continua o mesmo
class AudioRequest(BaseModel):
    audio_url: str

# Novo modelo de saída, com PDF e transcrição
class ProcessAudioResponse(BaseModel):
    pdf_url: str
    meeting_transcription: str

app = FastAPI()

@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: AudioRequest):
    """
    Endpoint para processar o áudio e retornar a URL do PDF gerado
    e a transcrição completa da reunião.
    """
    try:
        result = graph.invoke({
            "audio_file": request.audio_url
        })

        pdf_url = result.get("pdf_url")
        if not pdf_url:
            raise HTTPException(status_code=500, detail="Erro ao gerar o PDF.")

        # Extrai a transcrição produzida em transcribe_audio()
        meeting_transcription = result.get("meeting_transcript", "")

        return ProcessAudioResponse(
            pdf_url=pdf_url,
            meeting_transcription=meeting_transcription
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o áudio: {e}")