import os
import json
from collections import defaultdict
import cv2
import numpy as np

from Entidades.preprocessamento import PreProcessamento
from Entidades.analisespray import AnaliseSpray
from Entidades.visualizacao import Visualizacao
from Entidades.exportacao import Exportacao

def carregar_config(caminho_config="config.json"):
    with open(caminho_config, 'r') as f:
        return json.load(f)

def processar_captura(num_captura, config, pre_proc, analisador, visualizador):
    """Processa uma captura completa"""
    pasta_base = config["nome_pasta"]
    pasta_captura = os.path.join(pasta_base, f"captura{num_captura}")
    arquivo_tif = os.path.join(pasta_captura, f"captura{num_captura}.tif")

    try:
        imagens_por_ciclo = config["imagens_por_ciclo"][str(num_captura)]
    except KeyError:
        print(f"\n'imagens_por_ciclo' nao configurado para captura {num_captura}. Pulando.")
        return None

    imagens = pre_proc.carregar_imagens_tif(arquivo_tif)
    if imagens is None:
        return None

    # Pre processamento
    sem_spray = pre_proc.criar_imagem_fundo(imagens, config["num_injecoes"], imagens_por_ciclo)
    pre_proc.subtrair_fundo(
        imagens, sem_spray, config["inicio"], config["num_injecoes"],
        imagens_por_ciclo, config["frames_por_injecao"], pasta_captura, num_captura
    )

    # Analise de cada frame
    print(f"\nAnalisando frames da captura {num_captura}...")
    
    penetracoes = defaultdict(dict)
    angulos_cone = defaultdict(dict)
    desvios = defaultdict(dict)

    arquivos_pgm = sorted([f for f in os.listdir(pasta_captura) if f.startswith(f"captura{num_captura}") and f.endswith(".pgm")])

    for arquivo in arquivos_pgm:
        caminho_completo = os.path.join(pasta_captura, arquivo)
        partes = arquivo.replace('.pgm', '').split('_')
        injecao = int(partes[1][3:])
        frame = int(partes[2][5:])

        dados_bordas, img_com_bordas = analisador.detectar_bordas(caminho_completo)
        
        # Salva imagem com contorno desenhado
        cv2.imwrite(caminho_completo, img_com_bordas)

        if not dados_bordas:
            print(f"  Aviso: Nenhum contorno encontrado em {arquivo}")
            continue

        # Calcula os parametros
        penetracoes[injecao][frame] = analisador.calcular_penetracao(dados_bordas)
        angulo, desvio, m_e, m_d = analisador.calcular_angulo_desvio(dados_bordas, frame)
        angulos_cone[injecao][frame] = angulo
        desvios[injecao][frame] = desvio

        print(f"  {arquivo}: Pen: {penetracoes[injecao][frame]:.3f} cm | Ang: {angulo if angulo else 'N/A'} | Desv: {desvio if desvio else 'N/A'}")

        # Visualizacao
        if config["salvar_visualizacoes"] and m_e is not None and m_d is not None:
            caminho_saida_vis = os.path.join(pasta_captura, "visualizacoes")
            os.makedirs(caminho_saida_vis, exist_ok=True)
            
            idx_original = config["inicio"] + imagens_por_ciclo * (injecao - 1) + (frame - 1)
            img_original = imagens[idx_original] if idx_original < len(imagens) else np.zeros_like(img_com_bordas)
            
            visualizador.salvar_visualizacao_retas(
                img_original, img_com_bordas, m_e, m_d,
                caminho_saida_vis, partes[-1], dados_bordas
            )
            
    return {"penetracoes": penetracoes, "angulos_cone": angulos_cone, "desvios": desvios}

def main():
    config = carregar_config()

    pre_processador = PreProcessamento(config["primeira_imagem_com_spray"], config["num_imagens_subtracao"])
    analisador_spray = AnaliseSpray(config["pxcm"], config["corte"])
    visualizador = Visualizacao(config["pxcm"])
    
    diretorio_saida_excel = os.path.join("resultados_excel", f"dados_{config['nome_pasta']}")
    exportador = Exportacao(config["fps"], diretorio_saida_excel)

    for i in range(1, config["num_capturas"] + 1):
        resultados = processar_captura(i, config, pre_processador, analisador_spray, visualizador)
        if resultados:
            exportador.adicionar_resultados_captura(
                i, 
                resultados["penetracoes"], 
                resultados["angulos_cone"], 
                resultados["desvios"]
            )

    exportador.exportar_para_excel()

main()