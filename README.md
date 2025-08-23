# Descrição
Esse código foi desenvolvido para realizar a análise de sprays de combustível do laboratório de velocimetria do CTM - UFMG. Ele tem como resultado arquivos Excel com penetrações, ângulos de cone e desvios através do processamento de imagens TIFF e detecção de bordas.

# Bibliotecas necessárias
pip install numpy scikit-learn opencv-python pandas tifffile matplotlib

# Configurações
Para realizar o processamento de uma pasta de arquivos eles precisam estar organizados da seguinte forma:
- pasta/
  - captura1/  
    - captura1.tif  
  - captura2/  
    - captura2.tif  
  - ...  
- Entidades/...
- config.json  
- main.py  

A visualização permite que você valide o ângulo de cone calculado, caso queira que as imagens de validação sejam criadas ative salvar_visualizacoes no config.json;
Altere o arquivo config.json para os parâmetros do ensaio em questão.

# Ajustes

Alterar pxcm para a escala da câmera;  
Aumentar o valor de corte para diminuir a sensibilidade da detecção de bordas;


