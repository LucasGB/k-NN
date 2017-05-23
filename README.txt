1 - Instale os modulos utilizando os comandos:

python get-pip.py
pip install -U scipy.stats
pip install -U scikit-learn

2 - Copie os arquivos de treino e testes para o mesmo diretório do script

Para executar o script, digite no terminal o comando no seguinte formato:

python knn.py <arquivo_treino> <arquivo_teste> <k> <euclidiana/manhattan> <minmax/zscore>