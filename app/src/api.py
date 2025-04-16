from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import graph

# Define o modelo de entrada para a API
class AudioRequest(BaseModel):
    audio_url: str

# Define o modelo de saída para a API
class PDFResponse(BaseModel):
    pdf_url: str

# Inicializa a aplicação FastAPI
app = FastAPI()

@app.post("/process-audio", response_model=PDFResponse)
async def process_audio(request: AudioRequest):
    """
    Endpoint para processar o áudio e retornar a URL do PDF gerado.
    """
    try:
        # Executa o grafo com a URL do áudio
        result = graph.invoke({
            "audio_file": request.audio_url
        })

        # Obtém a URL do PDF gerado
        pdf_url = result.get("pdf_url")
        if not pdf_url:
            raise HTTPException(status_code=500, detail="Erro ao gerar o PDF.")

        return PDFResponse(pdf_url=pdf_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o áudio: {str(e)}")