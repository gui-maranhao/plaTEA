# plaTEA 

plaTEA é uma aplicação capaz de reconhecer a emoção detectada através da análise da fala e da face do usuário. Ela foi idealizada para ser utilizada em reuniões virtuais, a fim de auxiliar usuários que estão no espectro autista.

Esse projeto possui como base o [EmotionRAM](https://gitcin.cin.ufpe.br/voxarlabs/emotionram_faces_demo), trabalho realizado pelo Voxar Labs.

## Uso

Para utilização, olhe o arquivo `installinstructions.txt` para os comandos de instalação dos pacotes. Recomenda-se que seja utilizado um ambiente virtual Anaconda para isso. Então, basta rodar o arquivo `webcam.py` no seu terminal.

É necessário também que exista uma webcam virtual disponível no computador do usuário, para que seja sincronizada com os ambientes de reuniões virtuais.

## Instruções de setup

```Bash
# Para criar e ativar um ambiente virtual
conda create -n platea python=3.8 -y
conda activate platea

# Instalar os pacotes requeridos
pip install -r requirements.txt
```

### Rodando a aplicação

```Bash
# Usando GPU (MPS on Mac, CUDA on others)
python webcam.py --device gpu

# Ou forçando a utilização da CPU
python webcam.py --device cpu

# Para utilização de uma camera diferente, especifique o ID
python webcam.py --cameraID 1
```

Após isso, basta sincronizar com o seu ambiente de reunião virtual.
