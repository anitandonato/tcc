import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

CSV_PATH = "results/performance_summary.csv"
CHART_SAVE_DIR = "results/charts"

def plot_combined_final_chart():
    """
    Lê o CSV e gera UM gráfico compacto (proporção 41.5:44.8) de eixo duplo.
    - Eixo Y Esquerdo (Barras): Tempo Médio (ms) em ESCALA LOGARÍTMICA.
    - Eixo Y Direito (Linha): Faces Encontradas.
    - O FPS é ignorado por ser redundante (1/Tempo).
    """
    
    # --- 1. Ler e Preparar os Dados ---
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{CSV_PATH}' não foi encontrado.")
        print("Por favor, execute 'python tests/performance_tests.py' primeiro.")
        return
        
    df = df[df["FPS"] != "ERRO"].copy()
    
    df["Tempo Medio (ms)"] = pd.to_numeric(df["Tempo Medio (ms)"])
    df["Faces Encontradas"] = pd.to_numeric(df["Faces Encontradas"])
    
    # Ordena pelo Tempo (do mais rápido ao mais lento)
    df_sorted = df.sort_values(by="Tempo Medio (ms)", ascending=True)
    
    os.makedirs(CHART_SAVE_DIR, exist_ok=True)
    print(f"Lendo dados de '{CSV_PATH}' e salvando gráfico final em '{CHART_SAVE_DIR}'...")

    # --- 2. Criar o Gráfico de Eixo Duplo ---
    
    # Tamanho = Rácio de 41.5 / 44.8 ~= 0.926 (Largura/Altura)
    # figsize=(8, 8.64) é (800x864 pixels) - Respeita o seu rácio
    fig, ax1 = plt.subplots(figsize=(8, 8.64)) 
    
    cor_barra = 'skyblue'
    ax1.set_title('Performance: Custo (Log-Tempo) vs. Resultado (Faces)', fontsize=16)
    ax1.set_xlabel('Biblioteca', fontsize=12)
    ax1.set_ylabel('Custo: Tempo Médio (ms) - [Escala Logarítmica]', color=cor_barra, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=cor_barra)
    
    # Plotar as barras de Tempo
    bars = ax1.bar(df_sorted["Biblioteca"], df_sorted["Tempo Medio (ms)"], color=cor_barra, label='Tempo (ms)')
    
    # *** A SOLUÇÃO: ESCALA LOGARÍTMICA ***
    # Isto "comprime" as barras grandes e "expande" as pequenas
    ax1.set_yscale('log')
    
    # Adiciona os rótulos (labels) nas barras
    ax1.bar_label(bars, fmt='%.0f ms', fontsize=9) 

    # Rotação dos labels do eixo X (tem que vir *depois* do bar())
    ax1.set_xticklabels(df_sorted["Biblioteca"], rotation=45, ha="right", fontsize=10)

    # Criar o segundo eixo (ax2) que partilha o eixo X
    ax2 = ax1.twinx() 
    
    cor_linha = 'salmon'
    ax2.set_ylabel('Resultado: Faces Encontradas', color=cor_linha, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=cor_linha)
    
    # Plotar a Linha de Faces Encontradas
    line = ax2.plot(df_sorted["Biblioteca"], df_sorted["Faces Encontradas"], 
                    color=cor_linha, marker='o', linewidth=2, label='Faces Encontradas')

    # Ajusta o eixo Y da linha para ser limpo (0, 1, 2, 3)
    ax2.set_yticks(np.arange(0, df_sorted["Faces Encontradas"].max() + 2, step=1)) 
    
    # Adicionar os valores na linha (pontos)
    for i, txt in enumerate(df_sorted["Faces Encontradas"]):
        ax2.annotate(txt, (df_sorted["Biblioteca"].iloc[i], df_sorted["Faces Encontradas"].iloc[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', color=cor_linha)

    # --- 3. Salvar o Gráfico ---
    plt.tight_layout() # Ajusta tudo para caber na imagem
    save_path_combined = os.path.join(CHART_SAVE_DIR, "performance_combinada_final.png")
    plt.savefig(save_path_combined, dpi=100) # dpi=100 para 800x864 pixels
    
    print(f"\nGráfico final salvo em: '{save_path_combined}'")

if __name__ == "__main__":
    plot_combined_final_chart()