#!/bin/sh
## Script Para Rodar o Logistic Regression
########################################################################################################
### Versão corrigida e comentada de acordo com o Administrador de Redes do LIS
#########################################################################################################

# Maintainer: Pedro Assis, Larissa Feliciano e Igor Rodrigues <pedrojudo74@hotmail.com>
# Description: WebServer
# Date: 16/01/2019
# Start script sudo ./script_crossdomain_larissa.sh


############################################### STEP 01
## echo "Update Package.."
##  sudo apt-get update -y && sudo apt-get upgrade -y && sudo apt-get clean -y  // Não se pode usar o comando sudo

############################################### STEP 02
# echo "Install Dependencies.;."

## echo "Install Git"
## sudo apt install git -y  /// não precisa porque já está instalado


############################################### STEP 03
echo "Install Dependencies Python Project Requirements..."
conda create --name cross_domain_env python=3.6 && \
source activate cross_domain_env && \
conda install -c conda-forge spacy=2.0.11 && \
python -m spacy download en && \
conda instal matplotlib=3.0.2 &&\
conda install -c anaconda scikit-learn=0.20.2 && \
conda install -c menpo pathlib && \
conda install -c conda-forge flask-socketio && \  
conda install -c anaconda six=1.11.0 && \ 
conda install -c anaconda numpy=1.14.2 && \
conda install -c anaconda pickle &&\
###### Novas bibliotecas adicionadas ######
conda install _tflow_select=2.1.0-gpu
conda install tensorflow=1.12.0 &&\
conda install keras=2.2.4 &&\
conda install pandas=0.23.4 &&\
conda install -c anaconda nltk=3.3 &&\
python -m nltk.downloader all &&\
conda install -c anaconda scipy=1.2.0 &&\
conda install -c opencog-ull sparsesvd=0.2.2 
pip install swarmpackagepy=1.0.0a5
#####  FALTA INFORMAÇÃO AQUI DO NÚMERO DE VERSÃO DAS BIBLIOTECAS DO PYTHON
### Listar as bibliotecas que você tem para este enviroment e precisar no comando "conda install ..."
### o número de versão de cada biblioteca a instalar
########################################################################################################


############################################### STEP 05
echo "Download Project.."
git clone https://github.com/IgorSouza21/Cross_Domain
echo "Install Completed!"


############################################### STEP 06
echo "Run Application!"

# Entra no diretório do seu projeto clonado
cd Cross_Domain

# executa o arquivo principal do seu projeto
python main.py lr
