#run.py
from dotenv import load_dotenv
from src.agent import graph

# Carrega variáveis de ambiente
load_dotenv()

def main():
    # Transcrição de exemplo
    transcript = """
    Reunião de alinhamento do projeto de desenvolvimento de software.
    Participantes: João (Gerente de Projeto), Maria (Desenvolvedora), Pedro (Designer)

    João: Precisamos discutir o andamento do projeto de CRM.
    Maria: Já completei 70% do backend.
    Pedro: O design das interfaces está 50% concluído.
    João: Vamos definir os próximos passos...
    """

    # Executa o grafo
    result = graph.invoke({
        "meeting_transcript": transcript
    })

    print("Resumo da Reunião:")
    print(result['meeting_summary'])

if __name__ == "__main__":
    main()
