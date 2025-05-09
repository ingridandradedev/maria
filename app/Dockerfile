# filepath: d:\PROJETOS  VS CODE\maria\app\Dockerfile

# Use uma imagem base do Python
FROM python:3.11-slim

# Instala dependências do sistema e o wkhtmltopdf
RUN apt-get update && apt-get install -y \
    wkhtmltopdf \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    && apt-get clean

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de dependências
COPY requirements.txt . 

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY . .  

# Copia o arquivo de credenciais para o contêiner
COPY maria-457717-9fa8d402e552.json /app/maria-457717-9fa8d402e552.json

# Expõe a nova porta que o FastAPI usará
EXPOSE 8080

# Comando para iniciar a aplicação
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]