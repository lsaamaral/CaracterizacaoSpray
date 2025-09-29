import os
import tifffile
import json
from Entidades.preprocessamento import Preprocessamento
from Entidades.analisespray import AnaliseSpray
from Entidades.exportacao import Exportacao

def carregar_config(caminho_config):
    with open(caminho_config, 'r') as f:
        return json.load(f)

def main():
    config = carregar_config("config.json")

    preproc = Preprocessamento(pxcm=config["pxcm"], corte=config["corte"])
    analise = AnaliseSpray(pxcm=config["pxcm"], corte=config["corte"], salvar_visualizacoes=config["salvar_visualizacoes"])


    for captura_numero in range(1, config["num_capturas"] + 1):
        pasta_captura = os.path.join(config["nome_pasta"], f"captura{captura_numero}")
        arquivo_tif = os.path.join(pasta_captura, f"captura{captura_numero}.tif")

        if not os.path.exists(arquivo_tif):
            continue

        with tifffile.TiffFile(arquivo_tif) as tif:
            imagens = tif.asarray()

        sem_spray = preproc.criar_imagem_fundo(imagens, config["primeira_imagem_com_spray"],
                                               config["num_injecoes"], config["imagens_por_ciclo"][str(captura_numero)],
                                               config["num_imagens_criarfundo"])

        preproc.subtrair_fundo(imagens, sem_spray, config["inicio"], config["num_injecoes"],
                               config["imagens_por_ciclo"][str(captura_numero)], config["frames_por_injecao"],
                               pasta_captura, captura_numero)

        analise.processar_todas_bordas(pasta_captura, captura_numero, imagens,
                                       config["inicio"], config["imagens_por_ciclo"][str(captura_numero)])

    Exportacao.exportar_resultados(analise.penetracoes, analise.angulos_cone, analise.desvios,
                                   fps=config["fps"], saida_dir="resultados_excel")

main()