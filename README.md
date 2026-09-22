# Detecção em Tempo Real de Distrações em Motoristas

### Arquitetura multimodal baseada em YOLOv8 e MediaPipe para monitoramento simultâneo de distrações em hardware com recursos computacionais limitados.

## Sobre

Este projeto apresenta uma arquitetura baseada em **Inteligência Artificial** e **Visão Computacional** para detecção simultânea e em tempo real de distrações relacionadas ao **uso do telefone celular** e à **sonolência do motorista**.

A solução integra o **YOLOv8**, responsável pela detecção do telefone celular, ao **Google MediaPipe Face Mesh**, utilizado para rastreamento facial e análise do estado dos olhos.

Além da identificação das distrações, o sistema incorpora **alertas sonoros e visuais**, permitindo uma resposta imediata quando situações de risco são detectadas.

O estudo também investiga a viabilidade da execução dessa arquitetura em **hardware com recursos computacionais limitados**, avaliando o equilíbrio entre desempenho de classificação e eficiência computacional.

---

# Objetivo

Desenvolver e avaliar um sistema baseado em inteligência artificial para a detecção simultânea e em tempo real de distrações de motoristas, por meio da integração entre os modelos YOLOv8 e MediaPipe, com emissão de alertas sonoros e visuais.

---

# Arquitetura Proposta

A arquitetura integra dois módulos principais de Visão Computacional:

- **YOLOv8:** responsável pela detecção do telefone celular;
- **MediaPipe Face Mesh:** responsável pelo rastreamento facial e análise dos movimentos oculares.

Os frames capturados pela câmera são compartilhados entre os módulos de processamento, permitindo analisar simultaneamente diferentes tipos de distração.

Os resultados são encaminhados à lógica de decisão do sistema, responsável pela emissão dos alertas sonoros e visuais.

```text
                         Captura de vídeo
                                │
                                ▼
                       Pré-processamento
                                │
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

# Detecção de Celular

A detecção do telefone celular é realizada utilizando modelos da família **YOLOv8**.

O sistema monitora a presença do objeto celular no campo de visão da câmera e, quando uma detecção válida é identificada, ativa os mecanismos de alerta da aplicação.

Durante a pesquisa foram avaliadas as seguintes variantes:

- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLOv8l
- YOLOv8x

Essa comparação permite analisar o impacto da complexidade de cada modelo sobre a precisão e a velocidade de processamento.

---

# Detecção de Sonolência

A identificação de sinais de sonolência utiliza os landmarks faciais fornecidos pelo **MediaPipe Face Mesh**.

A abertura dos olhos é analisada por meio do **Eye Aspect Ratio (EAR)**, calculado a partir das distâncias entre pontos específicos das pálpebras.

Antes do monitoramento, o sistema realiza uma etapa de **calibração de 2,5 segundos**, utilizada para estabelecer valores de referência individuais para cada olho.

A partir desses valores são definidos limiares adaptativos de fechamento ocular.

Quando ambos os olhos permanecem abaixo dos respectivos limiares durante um período igual ou superior a **2 segundos**, o sistema caracteriza o evento como sonolência e emite o alerta correspondente.

---

# Processamento Concorrente

A arquitetura utiliza **threads independentes** para a execução dos mecanismos de alerta.

Essa abordagem evita que a reprodução dos alertas interrompa diretamente o pipeline principal de processamento das imagens.

A separação das tarefas contribui para reduzir bloqueios durante a execução e permite que os módulos de detecção continuem processando os frames enquanto os alertas são emitidos.

---

# Base de Dados Experimental

Foi desenvolvido um conjunto de dados próprio para avaliação da arquitetura.

A base é composta por:

- **120 vídeos**;
- **5 segundos por vídeo**;
- resolução de **1280 × 720 pixels**;
- aquisição a **30 FPS**;
- câmera posicionada aproximadamente **60 cm do condutor**;
- gravações em condições **diurnas, crepusculares e noturnas**;
- situações **com e sem óculos de grau**.

Os vídeos foram distribuídos de forma balanceada em quatro classes:

| Classe | Quantidade |
|---|---:|
| Normal | 30 vídeos |
| Detecção de Celular | 30 vídeos |
| Piscada Prolongada | 30 vídeos |
| Sonolência | 30 vídeos |
| **Total** | **120 vídeos** |

A construção de uma base própria permitiu controlar as condições de aquisição e avaliar conjuntamente estados relacionados à distração e à sonolência.

---

# Avaliação de Desempenho

A arquitetura foi avaliada utilizando diferentes variantes do YOLOv8 integradas ao MediaPipe, com execução em **CPU e GPU**.

As métricas utilizadas foram:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-score**
- **Frames Per Second (FPS)**

A análise considera não apenas a capacidade de classificação, mas também a eficiência computacional necessária para preservar a continuidade temporal do monitoramento.

---

# Resultados em GPU

| Modelo | Acurácia | Precisão | Recall | F1-score | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.898 | 0.894 | 0.890 | 0.892 | 29.2 |
| YOLOv8s | 0.925 | 0.920 | 0.917 | 0.918 | 25.8 |
| **YOLOv8m** | **0.948** | **0.945** | **0.943** | **0.944** | **20.5** |
| YOLOv8l | 0.970 | 0.967 | 0.965 | 0.966 | 17.2 |
| YOLOv8x | 0.983 | 0.980 | 0.978 | 0.979 | 14.6 |

Em GPU, o aumento da complexidade dos modelos proporciona ganhos progressivos nas métricas de classificação, acompanhado de redução da taxa de processamento.

O **YOLOv8m** apresenta uma relação favorável entre desempenho de classificação e eficiência computacional, mantendo taxa de processamento suficiente para preservar a análise temporal realizada pelo MediaPipe.

---

# Resultados em CPU

| Modelo | Acurácia | Precisão | Recall | F1-score | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.875 | 0.872 | 0.869 | 0.870 | 10.6 |
| YOLOv8s | 0.918 | 0.913 | 0.909 | 0.911 | 8.5 |
| **YOLOv8m** | **0.934** | **0.930** | **0.928** | **0.929** | **5.7** |
| YOLOv8l | 0.923 | 0.921 | 0.919 | 0.920 | 3.0 |
| YOLOv8x | 0.908 | 0.904 | 0.901 | 0.902 | 2.4 |

Em CPU, o aumento da complexidade do detector passa a reduzir significativamente a quantidade de frames processados.

Essa redução afeta principalmente eventos que dependem da continuidade temporal, como **Piscada Prolongada** e **Sonolência**.

---

# Sweet Spot

Um dos principais pontos investigados nesta pesquisa é o equilíbrio entre **desempenho de classificação** e **eficiência computacional**.

Os resultados indicam que o **YOLOv8m integrado ao MediaPipe** ocupa a região correspondente ao *sweet spot* da aplicação em hardware com recursos computacionais limitados.

Em CPU, essa combinação alcançou:

```text
Acurácia: 0.934
Precisão: 0.930
Recall:   0.928
F1-score: 0.929
FPS:      5.7
```

Os resultados demonstram que a escolha do modelo não deve considerar apenas a maior capacidade de classificação, mas também a quantidade de frames disponível para preservar a análise temporal realizada pelo MediaPipe.

---

# Ambiente Experimental

Os experimentos foram realizados utilizando:

```text
Sistema Operacional: Windows 11 - 64 bits
Python: 3.12.5
CPU: Intel Core i5-12500H
Memória RAM: 16 GB
GPU: NVIDIA GeForce RTX 3050 8 GB
CUDA: 11.8
Webcam: Logitech C922
```

Principais bibliotecas utilizadas:

- Python
- OpenCV
- Ultralytics YOLOv8
- Google MediaPipe
- NumPy
- Matplotlib
- Threading
- Winsound

---

# Funcionamento

O sistema executa continuamente as seguintes etapas:

```text
1. Captura dos frames da câmera
                │
                ▼
2. Processamento das imagens
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 Detecção celular    Análise facial
     YOLOv8          MediaPipe + EAR
        │                │
        └───────┬────────┘
                │
                ▼
3. Análise do estado do motorista
                │
                ▼
4. Identificação da distração
                │
                ▼
5. Emissão de alertas sonoros e visuais
```

O sistema pode apresentar diferentes estados durante o monitoramento, incluindo:

```text
ACORDADO
SEM CELULAR
ALERTA SONO
ALERTA CELULAR
```

---

# Principais Contribuições

- Integração entre **YOLOv8 e MediaPipe** em uma única arquitetura.
- Detecção simultânea do uso do celular e de sinais de sonolência.
- Utilização de **EAR com calibração individual** para análise do fechamento ocular.
- Arquitetura com processamento e alertas organizados de forma concorrente.
- Avaliação das variantes YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l e YOLOv8x.
- Comparação experimental entre execução em CPU e GPU.
- Avaliação utilizando Acurácia, Precisão, Recall, F1-score e FPS.
- Investigação do equilíbrio entre desempenho de classificação e eficiência computacional.
- Identificação do **sweet spot** para execução em hardware com recursos computacionais limitados.
- Validação da arquitetura em ambiente veicular.

---

# Aplicações

A arquitetura pode ser utilizada como base para estudos e aplicações relacionadas a:

- monitoramento do estado de alerta do motorista;
- sistemas de apoio à condução;
- detecção de distrações em veículos;
- análise de sonolência por Visão Computacional;
- aplicações de Inteligência Artificial em hardware limitado;
- avaliação de desempenho de modelos de detecção em CPU e GPU;
- sistemas de segurança veicular em tempo real.

---

# Estrutura do Projeto

```text
driver-distraction-detection/

├── images/
├── src/
├── README.md
└── requirements.txt
```

---
