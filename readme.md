# Decision Tree Generator using ID3 Algorithm

### Autores
- Pedro Campião
- Guilherme Vaz
- Ricardo Costa

Terceiro trabalho prático da cadeira de 
Inteligência Artificial da Licenciatura em
Inteligência Artifical e Ciência de Dados da
Faculdade de Ciências da Universidade do Porto.

Este programa treina uma Decision Tree 
com base num dataset guardado como csv, 
utilizando o algoritmo ID3. É possível também
testar a árvore com um novo dataset de teste
com novas entradas.

O código foi desenvolvido e testado utilizando
o seguinte sistema operativo e versão do Python:
- Windows 11 22H2
- Python 3.10.11

As seguintes bibliotecas foram utilizadas e 
devem estar instaladas para o programa correr:
- Numpy
- Pandas 


Para correr o programa de modo a treinar a 
árvore e executar testes na árvore criada,
corra o programa da seguinte forma:

```bash
python3 main.py -e train_file.csv -t test_file.csv
```

Caso haja alguma dúvida de como executar
o programa, corra apenas `python3 main.py -h`
para mostrar a seguinte ajuda:
```
usage: Decision Tree Generator using ID3 Algorithm [-h] [-e EXAMPLES] [-t TESTS]

options:
  -h, --help            show this help message and exit
  -e EXAMPLES, --examples EXAMPLES
                        CSV file name to train the learning tree
  -t TESTS, --tests TESTS
                        CSV file name to test the learning tree obtained
```




