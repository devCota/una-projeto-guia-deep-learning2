# Classificação de Imagens com Deep Learning — fastai, PyTorch e Gradio

![Status](https://img.shields.io/badge/status-concluído-green)
![Plataforma](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20Colab-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Deep Learning](https://img.shields.io/badge/deep%20learning-fastai%20%7C%20PyTorch-orange)
![Interface](https://img.shields.io/badge/interface-Gradio-ff69b4)
![Modelo](https://img.shields.io/badge/modelo-ResNet34-lightgrey)
![Licença](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Sobre o Projeto

Este repositório apresenta um projeto prático de **Deep Learning aplicado à classificação de imagens**, desenvolvido com foco acadêmico e inspirado no Capítulo 3 do livro **Deep Learning for Coders with fastai and PyTorch**, de Jeremy Howard.

O projeto também segue a proposta prática do **GUIA — projeto extracurricular do Professor Leonardo**, do **Centro Universitário UNA**, com o objetivo de demonstrar um fluxo completo de Machine Learning:

1. preparar os dados;
2. treinar um modelo;
3. interpretar os resultados;
4. exportar o modelo treinado;
5. utilizar o modelo em uma interface gráfica simples.

A aplicação utiliza o dataset **Oxford-IIIT Pets**, baixado automaticamente pelo fastai, para treinar um classificador binário capaz de reconhecer:

- `gato`
- `cachorro`

Após o treinamento, o modelo é exportado para `models/pet_classifier.pkl` e pode ser usado em uma interface web criada com **Gradio**, onde o usuário envia uma imagem e recebe a previsão com o nível de confiança.

---

## Objetivos da Atividade

- Aplicar conceitos fundamentais de Deep Learning.
- Trabalhar com classificação de imagens em Computer Vision.
- Utilizar modelos pré-treinados com Transfer Learning.
- Construir um pipeline completo de Machine Learning.
- Treinar, interpretar, exportar e reutilizar um modelo.
- Criar uma interface gráfica simples para uso prático do classificador.
- Demonstrar o projeto de forma clara em uma apresentação acadêmica.

---

## Conceitos Aplicados

- Deep Learning
- Computer Vision
- Transfer Learning
- Data Augmentation
- Treinamento supervisionado
- Avaliação de modelos
- Exportação de modelo treinado
- Inferência com modelo exportado
- Interface gráfica com Gradio

---

## Conteúdo Prático Desenvolvido

O projeto contempla um fluxo completo de construção e uso de um modelo de classificação de imagens:

1. Download automático do dataset Oxford-IIIT Pets.
2. Criação dos rótulos `gato` e `cachorro`.
3. Organização dos dados com `DataBlock` e `DataLoaders` do fastai.
4. Aplicação de transformações e Data Augmentation.
5. Treinamento com Transfer Learning usando `ResNet34`.
6. Avaliação com a métrica `error_rate`.
7. Geração de gráficos de interpretação do modelo.
8. Exportação do modelo treinado em formato `.pkl`.
9. Carregamento do modelo exportado em uma interface Gradio.
10. Predição de novas imagens enviadas pelo usuário.

---

## Estrutura do Projeto

```text
ProjetoGuiaDeepLearning/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── treinar_modelo.py
├── models/
│   └── pet_classifier.pkl
├── reports/
│   ├── matriz_confusao.png
│   └── maiores_erros.png
└── data/
    └── oxford-iiit-pet/
```

### Descrição dos arquivos

- `src/treinar_modelo.py` → script responsável por baixar os dados, preparar os `DataLoaders`, treinar, interpretar e exportar o modelo.
- `app.py` → aplicação Gradio que carrega o modelo exportado e permite classificar imagens pela interface gráfica.
- `requirements.txt` → lista das dependências necessárias para executar o projeto.
- `models/pet_classifier.pkl` → modelo treinado exportado pelo fastai.
- `reports/` → pasta onde são salvos gráficos como matriz de confusão e maiores erros.
- `data/` → pasta local onde o dataset é armazenado após o download.
- `.gitignore` → configuração para evitar versionar ambiente virtual, cache, dataset, gráficos e modelo gerado.

---

## Modelo Utilizado

- Arquitetura: `ResNet34`
- Biblioteca principal: `fastai`
- Backend de Deep Learning: `PyTorch`
- Estratégia: **Transfer Learning**

O modelo parte de uma rede já treinada em grande escala, como no ImageNet, e é ajustado para o problema específico de classificação entre gatos e cachorros.

Essa abordagem reduz o custo computacional, diminui o tempo de treinamento e costuma gerar bons resultados mesmo com menos dados.

---

## Dataset Utilizado

O projeto utiliza o **Oxford-IIIT Pets Dataset**, acessado diretamente pelo fastai com:

```python
untar_data(URLs.PETS)
```

No dataset original, existe uma convenção nos nomes dos arquivos:

- nomes iniciados com letra maiúscula representam gatos;
- nomes iniciados com letra minúscula representam cachorros.

O script usa essa regra para criar automaticamente os rótulos das imagens.

---

## Métricas Utilizadas

Durante o treinamento, são avaliadas métricas como:

- `train_loss` → erro calculado nos dados de treino.
- `valid_loss` → erro calculado nos dados de validação.
- `error_rate` → taxa de erro do modelo.

Essas métricas ajudam a verificar se o modelo está aprendendo e se consegue generalizar para imagens que não foram usadas diretamente no treinamento.

---

## Dependências

As principais dependências do projeto são:

- Python 3.10 ou superior
- fastai
- PyTorch
- torchvision
- Gradio
- matplotlib
- Pillow
- IPython

Todas estão listadas no arquivo `requirements.txt`.

---

## Como Executar

### 1. Criar o ambiente virtual

No Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Treinar o modelo

```bash
python src/treinar_modelo.py
```

Para um treino mais rápido de demonstração:

```bash
python src/treinar_modelo.py --epochs 1 --skip-interpretation --max-images 200
```

### 4. Executar a interface gráfica

```bash
python app.py
```

Depois, acesse no navegador:

```text
http://127.0.0.1:7860
```

Caso a porta `7860` esteja ocupada, use outra porta:

```bash
python app.py --server-port 7862
```

---

## Execução no Google Colab

No Google Colab, instale as dependências com:

```python
!pip install fastai gradio matplotlib pillow ipython
```

Depois execute o treinamento:

```python
!python src/treinar_modelo.py --epochs 1 --skip-interpretation --max-images 200
```

Para abrir a interface com link público temporário:

```python
!python app.py --share --server-name 0.0.0.0
```

---

## Modelo Treinado

O modelo é exportado automaticamente como:

```text
models/pet_classifier.pkl
```

Esse arquivo contém:

- pesos treinados da rede neural;
- arquitetura e pipeline de inferência;
- transformações usadas pelo fastai;
- vocabulário das classes;
- informações necessárias para reutilizar o modelo.

Depois que o arquivo `.pkl` é gerado, não é necessário treinar novamente para usar a interface.

---

## Como Utilizar o Modelo Exportado

Exemplo simples de uso do modelo em Python:

```python
from fastai.vision.all import load_learner

learn = load_learner("models/pet_classifier.pkl")

pred, idx, probs = learn.predict("imagem.jpg")

print(pred)
print(probs)
```

A interface `app.py` usa a mesma ideia: ela carrega o arquivo `.pkl`, recebe uma imagem enviada pelo usuário e retorna as probabilidades de cada classe.

---

## Resultados de Exemplo

Exemplo de predição do modelo:

```python
('cachorro', tensor(0), tensor([0.8544, 0.1456]))
```

Interpretação:

- o modelo classificou a imagem como `cachorro`;
- a confiança aproximada foi de `85%`;
- a outra probabilidade corresponde à classe `gato`.

Os resultados podem variar conforme a quantidade de épocas, tamanho do dataset usado no treino, qualidade das imagens e presença ou não de GPU.

---

## Interface Gráfica

A interface foi construída com **Gradio**.

Fluxo de uso:

1. o usuário abre a página local;
2. envia uma imagem;
3. o modelo processa a imagem;
4. a interface exibe a classe prevista;
5. a interface também mostra a confiança da previsão.

Essa etapa demonstra como um modelo treinado pode sair do ambiente de desenvolvimento e ser usado por uma pessoa sem precisar escrever código.

---

## Observações

- O treinamento completo pode demorar em máquinas sem GPU.
- O comando com `--max-images` serve para demonstrações rápidas.
- Um modelo treinado com poucas imagens funciona para validar o fluxo, mas pode ter desempenho limitado.
- Para uma apresentação mais robusta, é recomendado treinar com mais imagens e mais épocas.
- O arquivo `.pkl` deve ser carregado apenas quando a origem for confiável, pois o fastai utiliza o mecanismo de pickle do Python.

---

## Boas Práticas Aplicadas

- Separação entre script de treinamento e aplicação de inferência.
- Uso de ambiente virtual isolado.
- Organização de pastas para modelos, dados e relatórios.
- Uso de Transfer Learning para reduzir custo computacional.
- Exportação do modelo para reutilização posterior.
- Interface simples e objetiva para uso prático.
- Código comentado e adequado para apresentação acadêmica.

---

## Possíveis Melhorias Futuras

- Classificar as 37 raças do Oxford-IIIT Pets em vez de apenas `gato` e `cachorro`.
- Adicionar exemplos prontos diretamente na interface Gradio.
- Mostrar a matriz de confusão dentro da própria interface.
- Criar uma versão em notebook para facilitar a apresentação no Google Colab.
- Comparar arquiteturas como `resnet18`, `resnet34` e `resnet50`.
- Fazer deploy do projeto em Hugging Face Spaces.

---

## Aplicação no Contexto Acadêmico

Este projeto reforça conceitos importantes para a formação em tecnologia:

- raciocínio lógico aplicado a dados;
- introdução ao aprendizado de máquina;
- manipulação de datasets reais;
- construção de pipelines de Machine Learning;
- interpretação de métricas de desempenho;
- uso de ferramentas modernas do mercado;
- aplicação prática de modelos de Deep Learning.

Essas competências são base para áreas como Inteligência Artificial, Ciência de Dados, Engenharia de Machine Learning e Visão Computacional.

---

## Referência Bibliográfica

Este projeto foi inspirado e fundamentado no conteúdo do livro:

**Deep Learning for Coders with fastai and PyTorch** — Jeremy Howard

A obra apresenta uma abordagem prática e acessível para construção de modelos de Deep Learning, sendo a base conceitual para a implementação deste projeto.

---

## Licença

Este projeto está licenciado sob a licença MIT.

---

## Conclusão

O projeto apresenta uma implementação funcional de classificação de imagens utilizando Deep Learning, com foco em clareza, organização e aplicação prática dos conceitos.

A solução demonstra o ciclo completo de um modelo de visão computacional: preparação dos dados, treinamento, avaliação, exportação e uso em uma interface gráfica.

---

## Autor

**Lucas Cota**
Estudante de Análise e Desenvolvimento de Sistemas
Foco em Backend e Engenharia de Software
