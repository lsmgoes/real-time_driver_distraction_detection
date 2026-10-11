# Detecção em Tempo Real de Distrações em Motoristas

### Arquitetura multimodal baseada em YOLOv8 e MediaPipe para monitoramento simultâneo de distrações em hardware com recursos computacionais limitados.

---

## Sobre

Este projeto apresenta uma arquitetura baseada em **Inteligência Artificial** e **Visão Computacional** para detecção simultânea e em tempo real de distrações relacionadas ao **uso do telefone celular** e à **sonolência do motorista**.

A solução integra o **YOLOv8**, responsável pela detecção do telefone celular, ao **Google MediaPipe Face Mesh**, utilizado para rastreamento facial e análise do estado dos olhos.

Além da identificação das distrações, o sistema incorpora **alertas sonoros e visuais**, permitindo sinalizar situações detectadas durante o monitoramento.

O projeto também disponibiliza um módulo específico para avaliação experimental das diferentes variantes do YOLOv8 integradas ao MediaPipe, permitindo calcular métricas de classificação e de desempenho computacional.

---

## Arquitetura Proposta

A arquitetura integra dois módulos principais de Visão Computacional:

- **YOLOv8:** responsável pela detecção do telefone celular;
- **MediaPipe Face Mesh:** responsável pelo rastreamento facial e análise dos movimentos oculares.

Os frames capturados pela câmera são processados pelos dois módulos de detecção.

Os resultados são encaminhados para a lógica de decisão do sistema, responsável pela definição do estado monitorado e pela emissão dos alertas correspondentes.

```text
                         Captura de vídeo
                                │
                                ▼
                       Pré-processamento
                                │
                                │
                          Processamento
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
              YOLOv8                      MediaPipe
       Detecção de celular              Face Mesh + EAR
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                        Lógica de decisão
                                │
                                ▼
                    Alertas sonoros e visuais
```

---

## Detecção de Celular

A detecção do telefone celular é realizada utilizando modelos da família **YOLOv8**, por meio da biblioteca Ultralytics.

A classe correspondente ao telefone celular é monitorada durante o processamento dos frames. Quando uma detecção atende ao limiar de confiança configurado, o sistema identifica a presença do objeto e pode emitir o alerta correspondente.

As seguintes variantes do YOLOv8 são utilizadas no módulo de avaliação:

- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLOv8l
- YOLOv8x

No módulo de execução em tempo real localizado em `detector/`, é utilizado o modelo **YOLOv8m**.

Os pesos dos modelos estão disponíveis na pasta:

```text
models/
```

---

## Detecção de Sonolência

A identificação de sinais relacionados à sonolência utiliza os landmarks faciais fornecidos pelo **MediaPipe Face Mesh**.

A abertura dos olhos é analisada por meio do **Eye Aspect Ratio (EAR)**, calculado a partir das distâncias entre pontos específicos das pálpebras.

Antes do monitoramento, o sistema realiza uma etapa de **calibração de 2,5 segundos**, utilizada para estabelecer valores de referência individuais para cada olho.

A partir desses valores são definidos limiares adaptativos de fechamento ocular.

O limiar é calculado a partir de **72% do EAR de referência**, respeitando os limites definidos no sistema.

Quando ambos os olhos permanecem abaixo dos respectivos limiares durante período igual ou superior a **2 segundos**, o sistema identifica a condição de sonolência e aciona o alerta correspondente.

Também são utilizados critérios de persistência temporal e suavização do EAR para reduzir alterações provocadas por variações isoladas entre frames consecutivos.

---

## Alertas Sonoros e Visuais

O sistema possui mecanismos de alerta visual e sonoro.

Durante a execução podem ser apresentados estados como:

```text
CALIBRANDO OLHOS
ACORDADO
SEM CELULAR
ALERTA SONO
ALERTA CELULAR
```

Os alertas sonoros são executados em **threads independentes**, evitando que sua reprodução interrompa diretamente o fluxo principal de processamento das imagens.

---

## Base de Dados Experimental

Foi desenvolvido um conjunto de dados próprio para avaliação da arquitetura proposta.

A base é composta por:

- **120 vídeos**;
- duração aproximada de **5 segundos por vídeo**;
- aquisição em ambiente veicular;
- câmera posicionada aproximadamente **60 cm do condutor**;
- gravações em condições **diurnas, crepusculares e noturnas**;
- situações **com e sem óculos de grau**.

Os vídeos estão distribuídos de forma balanceada em quatro classes:

| Classe | Quantidade |
|---|---:|
| Normal | 30 vídeos |
| Det_celular | 30 vídeos |
| Pisc_prolongada | 30 vídeos |
| Sonolencia | 30 vídeos |
| **Total** | **120 vídeos** |

A base está disponível em:

```text
avaliar_modelos/dataset/
```

Estrutura:

```text
dataset/
├── Det_celular/
├── Normal/
├── Pisc_prolongada/
└── Sonolencia/
```

### Utilização da base

Os modelos YOLOv8 utilizados neste projeto são **pré-treinados**.

A base experimental desenvolvida para esta pesquisa **não foi utilizada para treinamento ou ajuste dos pesos do YOLOv8**.

Dessa forma, não foi realizada divisão da base em conjuntos de treinamento, validação e teste.

Os **120 vídeos foram utilizados exclusivamente para avaliação experimental do sistema**, permitindo analisar as diferentes variantes do YOLOv8 integradas ao MediaPipe sob as mesmas condições de teste.

---

## Avaliação dos Modelos

O diretório:

```text
avaliar_modelos/
```

contém o script destinado à avaliação das seguintes variantes:

```text
YOLOv8n
YOLOv8s
YOLOv8m
YOLOv8l
YOLOv8x
```

Os modelos são avaliados sequencialmente utilizando a mesma base experimental.

As métricas calculadas são:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-score**
- **Frames Per Second (FPS)**

O módulo também permite gerar automaticamente:

- arquivos CSV com os resultados;
- métricas consolidadas;
- métricas por classe;
- matrizes de confusão;
- gráficos de Acurácia × FPS;
- tabela consolidada das métricas.

Os arquivos produzidos durante a execução são armazenados localmente em:

```text
avaliar_modelos/resultados/
```

A pasta de resultados não precisa ser mantida no repositório, pois seu conteúdo pode ser reproduzido executando novamente o módulo de avaliação.

---

## Modelos e Pesos

Os pesos utilizados estão disponíveis na pasta:

```text
models/
```

Estrutura:

```text
models/
├── yolov8n.pt
├── yolov8s.pt
├── yolov8m.pt
├── yolov8l.pt
└── yolov8x.pt
```

O módulo de detecção em tempo real utiliza:

```text
yolov8m.pt
```

O módulo de avaliação utiliza as cinco variantes disponibilizadas.

Os arquivos `.pt` são armazenados no repositório utilizando **Git LFS** devido ao tamanho dos modelos.

---

## Configurações Principais

Entre os principais parâmetros utilizados pelo sistema estão:

```text
Tempo de calibração do EAR: 2,5 s
Fator do limiar adaptativo do EAR: 0,72
Tempo para identificação de sonolência: 2,0 s
Confiança mínima para detecção de celular: 0,50
Suavização exponencial do EAR: α = 0,35
Classe YOLO utilizada para celular: 67
```

Outros parâmetros relacionados à resolução, região facial, pré-processamento, CLAHE, ajuste de luminosidade e processamento das imagens encontram-se definidos diretamente nos scripts.

---

## Ambiente Experimental

O ambiente utilizado durante os experimentos da pesquisa foi:

```text
Sistema Operacional: Windows 11 - 64 bits
Python: 3.12.5
CPU: Intel Core i5-12500H
Memória RAM: 16 GB
GPU: NVIDIA GeForce RTX 3050 8 GB
CUDA: 11.8
Webcam: Logitech C922
```

---

## Bibliotecas e Dependências

As principais bibliotecas utilizadas são:

- Python
- OpenCV
- Ultralytics YOLOv8
- Google MediaPipe
- NumPy
- Matplotlib
- Scikit-learn
- Threading
- Winsound

As dependências necessárias estão definidas separadamente em:

```text
detector/requirements.txt
avaliar_modelos/requirements.txt
```

O ambiente recomendado para execução é **Python 3.12**.

---

## Instalação

### 1. Instalar o Git LFS

O repositório utiliza **Git LFS** para armazenar os vídeos da base experimental e os pesos dos modelos.

Caso o Git LFS ainda não esteja instalado:

```bash
git lfs install
```

### 2. Clonar o repositório

```bash
git clone https://github.com/lsmgoes/real-time_driver_distraction_detection.git
```

Entre na pasta:

```bash
cd real-time_driver_distraction_detection
```

### 3. Baixar os arquivos armazenados no Git LFS

```bash
git lfs pull
```

Esse comando garante o download dos vídeos da base experimental e dos pesos dos modelos YOLOv8.

---

## Execução do Detector em Tempo Real

Entre na pasta:

```bash
cd detector
```

Crie um ambiente virtual utilizando Python 3.12:

```bash
py -3.12 -m venv .venv
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Execute o programa:

```bash
python main.py
```

Durante a execução, pressione:

```text
ESC
```

para encerrar o sistema.

---

## Execução da Avaliação dos Modelos

Entre na pasta:

```bash
cd avaliar_modelos
```

Crie o ambiente virtual:

```bash
py -3.12 -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

---

## Execução em CPU ou GPU

O dispositivo de processamento pode ser configurado no script de avaliação.

### CPU

```python
DEVICE = "cpu"
```

### GPU NVIDIA

```python
DEVICE = 0
```

Para utilização da GPU NVIDIA, o **PyTorch com suporte à CUDA** deve estar instalado de forma compatível com a GPU e com a versão do CUDA disponível na máquina.

As avaliações em CPU e GPU devem ser executadas separadamente.

---

## Estrutura do Projeto

```text
real-time_driver_distraction_detection/

├── README.md
├── .gitattributes
│
├── models/
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov8m.pt
│   ├── yolov8l.pt
│   └── yolov8x.pt
│
├── detector/
│   ├── main.py
│   └── requirements.txt
│
└── avaliar_modelos/
    ├── main.py
    ├── requirements.txt
    └── dataset/
        ├── Det_celular/
        ├── Normal/
        ├── Pisc_prolongada/
        └── Sonolencia/
```

---

## Organização dos Códigos

### `detector/main.py`

Contém a implementação utilizada para execução em tempo real, incluindo:

- captura da câmera;
- pré-processamento das imagens;
- integração entre YOLOv8 e MediaPipe;
- detecção do telefone celular;
- rastreamento facial;
- cálculo do EAR;
- calibração individual dos olhos;
- lógica temporal de fechamento ocular;
- alertas sonoros;
- alertas visuais.

### `avaliar_modelos/main.py`

Contém a implementação destinada à avaliação experimental, incluindo:

- leitura automática da base de vídeos;
- avaliação sequencial das variantes do YOLOv8;
- integração com MediaPipe e EAR;
- processamento em CPU ou GPU;
- cálculo das métricas;
- geração das matrizes de confusão;
- geração dos gráficos;
- exportação dos resultados em arquivos CSV.

---

## Arquivos de Configuração

O projeto utiliza os seguintes arquivos principais de configuração:

```text
detector/requirements.txt
avaliar_modelos/requirements.txt
.gitattributes
```

Os arquivos `requirements.txt` especificam as dependências necessárias para cada módulo.

O arquivo `.gitattributes` configura o uso do **Git LFS** para os vídeos `.mp4` e para os pesos `.pt`.

---

## Observações

O projeto foi desenvolvido prioritariamente para ambiente **Windows**.

O módulo de execução em tempo real utiliza recursos específicos desse sistema operacional, como:

- `winsound`;
- captura de câmera utilizando DirectShow.

Os arquivos de vídeo `.mp4` e os pesos dos modelos `.pt` são gerenciados por meio do **Git LFS**.

Os resultados experimentais obtidos durante a pesquisa não são apresentados neste README. Este repositório concentra os códigos, modelos, base experimental, dependências e instruções necessárias para reprodução da metodologia.
