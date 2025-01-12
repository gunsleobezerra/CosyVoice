import tkinter as tk
from tkinter import filedialog, messagebox
import torchaudio
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Função para carregar o modelo CosyVoice
def carregar_modelo():
    global cosyvoice
    try:
        cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
        messagebox.showinfo("Sucesso", "Modelo CosyVoice carregado com sucesso.")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao carregar o modelo: {e}")

# Função para selecionar o arquivo de áudio
def selecionar_audio():
    caminho_audio = filedialog.askopenfilename(filetypes=[("Arquivos de Áudio", "*.wav")])
    entrada_audio_var.set(caminho_audio)

# Função para clonar a voz e gerar o áudio
def clonar_voz():
    caminho_audio = entrada_audio_var.get()
    texto = entrada_texto.get("1.0", tk.END).strip()
    if not caminho_audio or not texto:
        messagebox.showwarning("Atenção", "Por favor, selecione um arquivo de áudio e insira o texto.")
        return

    try:
        # Carregar o áudio de referência
        audio_referencia = load_wav(caminho_audio, 16000)

        # Realizar a inferência
        resultados = cosyvoice.inference_zero_shot(texto, "", audio_referencia, stream=False)

        # Salvar o áudio gerado
        for i, resultado in enumerate(resultados):
            caminho_saida = f"saida_{i}.wav"
            torchaudio.save(caminho_saida, resultado['tts_speech'], cosyvoice.sample_rate)
            messagebox.showinfo("Sucesso", f"Áudio gerado e salvo como {caminho_saida}")

    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar o áudio: {e}")

# Configuração da interface gráfica
root = tk.Tk()
root.title("Clonagem de Voz com CosyVoice")

# Variáveis de entrada
entrada_audio_var = tk.StringVar()

# Layout
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Arquivo de Áudio de Referência:").grid(row=0, column=0, sticky="w")
tk.Entry(frame, textvariable=entrada_audio_var, width=50).grid(row=0, column=1, padx=5)
tk.Button(frame, text="Selecionar", command=selecionar_audio).grid(row=0, column=2)

tk.Label(frame, text="Texto para Síntese:").grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
entrada_texto = tk.Text(frame, width=60, height=10)
entrada_texto.grid(row=2, column=0, columnspan=3, pady=5)

tk.Button(frame, text="Carregar Modelo", command=carregar_modelo).grid(row=3, column=0, pady=10)
tk.Button(frame, text="Clonar Voz e Gerar Áudio", command=clonar_voz).grid(row=3, column=1, columnspan=2, pady=10)

root.mainloop()
