import subprocess
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

# CONFIGURAÇÕES
LINK_REUNIAO_ORIGINAL = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_ZDg3YWI3M2QtYzZhNC00ZjdhLWEwYjAtYjMxOTg0OWQ4OGI5%40thread.v2/0?context=%7b%22Tid%22%3a%22cbf42931-5cdd-43c3-97c9-47c2c1c4b0c3%22%2c%22Oid%22%3a%228e01e4d1-52b2-4870-be7a-34c1262ca17c%22%7d"
NOME_USUARIO = "GravadorBot"
DURACAO_MAXIMA = 60 * 60  # 1 hora
DISPOSITIVO_AUDIO = "Mixagem estéreo (Realtek(R) Audio)"  # Substitua conforme seu sistema

# GERA LINK MODO ANÔNIMO
def gerar_link_anonimo_direto(link_original):
    base = "https://teams.microsoft.com"
    path = link_original.replace(base, "")
    final_url = f"{base}/v2/?meetingjoin=true#{path}"
    if "anon=true" not in final_url:
        final_url += "&anon=true"
    if "deeplinkId=" not in final_url:
        final_url += "&deeplinkId=joinweb"
    return final_url

# INICIA GRAVAÇÃO COM FFMPEG
def iniciar_gravacao(nome_arquivo):
    comando = [
        r"C:\Users\Dell\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe",  # Caminho direto
        "-y",
        "-f", "dshow",
        "-i", f"audio={DISPOSITIVO_AUDIO}",
        "-acodec", "libmp3lame",
        nome_arquivo
    ]
    return subprocess.Popen(comando)

# FUNÇÃO PRINCIPAL
def entrar_e_gravar():
    LINK_REUNIAO = gerar_link_anonimo_direto(LINK_REUNIAO_ORIGINAL)
    print(f"📡 Abrindo reunião: {LINK_REUNIAO}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=300,
            args=[
                "--use-fake-ui-for-media-stream",
                "--allow-file-access-from-files"
                # ⚠️ REMOVIDO: "--use-fake-device-for-media-stream"
            ]
        )

        context = browser.new_context(
            permissions=["microphone", "camera"],
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )

        context.route("**/*", lambda route, req: route.abort() if req.url.startswith("msteams://") else route.continue_())

        page = context.new_page()
        page.on("console", lambda msg: print(f"[console::{msg.type}] {msg.text}"))
        page.on("close", lambda: print("⚠️ A aba foi fechada automaticamente pelo Teams."))

        page.goto(LINK_REUNIAO)

        # Preenche o nome
        try:
            page.wait_for_selector('[data-tid="prejoin-display-name-input"]', timeout=30000)
            page.fill('[data-tid="prejoin-display-name-input"]', NOME_USUARIO)
            print(f"✅ Nome preenchido como: {NOME_USUARIO}")
        except:
            print("❌ Campo de nome não encontrado.")

        time.sleep(3)

        # Desativa microfone
        try:
            mic_button = page.locator('[aria-label^="Microfone"]')
            if mic_button.get_attribute("aria-pressed") == "true":
                mic_button.click()
                print("🔇 Microfone desativado.")
            else:
                print("✔️ Microfone já estava desligado.")
        except:
            print("❌ Botão de microfone não encontrado.")

        time.sleep(2)

        # Desativa câmera
        try:
            cam_button = page.locator('[aria-label^="Câmera"]')
            if cam_button.get_attribute("aria-pressed") == "true":
                cam_button.click()
                print("📷 Câmera desativada.")
            else:
                print("✔️ Câmera já estava desligada.")
        except:
            print("❌ Botão de câmera não encontrado.")

        time.sleep(2)

        # Clica em "Ingressar agora"
        try:
            page.wait_for_selector('button:has-text("Ingressar agora")', timeout=20000)
            page.click('button:has-text("Ingressar agora")', force=True)
            print("🚪 Cliquei em 'Ingressar agora'")
        except:
            print("❌ Botão 'Ingressar agora' não encontrado.")

        # Detecta se está preso no lobby
        try:
            page.wait_for_selector("text=permitirá que você entre", timeout=15000)
            print("⏳ Preso no lobby. Aguardando autorização...")
            time.sleep(60)
        except:
            print("✅ Não detectado lobby — possivelmente entrou direto ou erro não apareceu.")

        time.sleep(10)

        # Inicia gravação
        nome_arquivo = f"gravacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        processo_ffmpeg = iniciar_gravacao(nome_arquivo)
        print(f"🎙️ Gravando áudio em: {nome_arquivo}")

        # Monitoramento
        tempo_inicio = time.time()
        while True:
            if (time.time() - tempo_inicio) > DURACAO_MAXIMA:
                print("⏱️ Tempo máximo de gravação atingido.")
                break
            time.sleep(5)

        processo_ffmpeg.terminate()
        browser.close()
        print(f"✅ Gravação finalizada: {nome_arquivo}")

# EXECUTA
if __name__ == "__main__":
    entrar_e_gravar()
