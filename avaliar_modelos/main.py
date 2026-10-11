# PREPARAÇÃO DO AMBIENTE - PRIMEIRA EXECUÇÃO
# Copiar e enviar no terminal do VScode:
#
# py -3.12 -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install -r requirements.txt

from pathlib import Path
import csv
import gc
import importlib.util
import math
import os
import platform
import sys
import time

# Reduz mensagens internas de bibliotecas no terminal.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_REQUIREMENTS = BASE_DIR / "requirements.txt"

MODULOS_OBRIGATORIOS = {
    "cv2": "opencv-contrib-python",
    "matplotlib": "matplotlib",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "ultralytics": "ultralytics",
}


def verificar_dependencias():
    faltando = [
        pacote
        for modulo, pacote in MODULOS_OBRIGATORIOS.items()
        if importlib.util.find_spec(modulo) is None
    ]

    if not faltando:
        return

    print("\nBibliotecas necessárias não encontradas:")

    for pacote in faltando:
        print(f"- {pacote}")

    print("\nCrie/ative um ambiente virtual e execute:")
    print("python -m pip install -r requirements.txt")

    if not ARQUIVO_REQUIREMENTS.exists():
        print(
            "\nO arquivo requirements.txt não foi encontrado "
            "na mesma pasta de main.py."
        )

    sys.exit(1)


verificar_dependencias()

try:
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt
    import mediapipe as mp
    import numpy as np
    import sklearn
    import torch
    import ultralytics

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    from ultralytics import YOLO

except Exception as erro:
    print("\nFalha ao carregar as bibliotecas do projeto.")
    print(f"Erro: {erro}")
    print("\nExecute novamente:")
    print("python -m pip install -r requirements.txt")
    sys.exit(1)


# CAMINHOS

PASTA_DATASET = BASE_DIR / "dataset"
PASTA_RESULTADOS = BASE_DIR / "resultados"


# DISPOSITIVO DE PROCESSAMENTO

# CPU:
DEVICE = "cpu"

# Para avaliar em GPU NVIDIA com CUDA, altere para:
# DEVICE = 0
#
# Antes de usar GPU, confirme no mesmo ambiente virtual:
#
# python -c "import torch; print(torch.cuda.is_available())"
#
# O resultado deve ser True.
#
# CPU e GPU devem ser avaliadas em execuções separadas
# para evitar interferência nas medições de FPS.


# MODELOS

MODELOS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]


# DATASET

CLASSES = [
    "Normal",
    "Det_celular",
    "Pisc_prolongada",
    "Sonolencia",
]

NOMES_EXIBICAO = {
    "Normal": "Normal",
    "Det_celular": "Det_celular",
    "Pisc_prolongada": "Pisc_prolongada",
    "Sonolencia": "Sonolência",
}

EXTENSOES_VIDEO = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}


# PROCESSAMENTO

RESOLUCAO_CAPTURA = (1280, 720)
RESOLUCAO_YOLO = (640, 480)

# Cada vídeo é lido primeiro para calibração individual do EAR.
# Depois ele é lido novamente, desde o início, para avaliação completa.
TEMPO_CALIBRACAO = 2.5

# Sono
TEMPO_FECHADO = 2.0
FRAMES_CONSECUTIVOS_FECHADO = 2

# Piscada prolongada
TEMPO_PISCADA_PROLONGADA = 0.5

# Celular
# Mantido igual ao detector original.
PULAR_FRAMES_YOLO = 1.5
CONFIANCA_CELULAR = 0.50
YOLO_IMGSZ = 320
MIN_DETECCOES_CELULAR = 1

# ROI facial
FATOR_EXPANSAO_LATERAL = 0.30
FATOR_EXPANSAO_SUPERIOR = 0.35
FATOR_EXPANSAO_INFERIOR = 0.15
FATOR_ZOOM_FACE = 3.0

MIN_LARGURA_FACE = 80
MIN_ALTURA_FACE = 80

# Pré-processamento
ATIVAR_CLAHE = True
ATIVAR_GAMMA_ADAPTIVO = True
GAMMA_MIN = 1.2
GAMMA_MAX = 1.8
BRILHO_REFERENCIA = 110

# EAR
ALPHA_EAR = 0.35
LIMIAR_EAR_MIN = 0.16
LIMIAR_EAR_MAX = 0.30
FATOR_LIMIAR_EAR = 0.72

# Valores de segurança usados somente se a calibração individual
# não conseguir obter nenhuma amostra facial válida.
EAR_BASE_FALLBACK = 0.24
EAR_LIMIAR_FALLBACK = 0.18


# LANDMARKS DOS OLHOS

P_DIR_H1 = 33
P_DIR_H2 = 133
P_DIR_V1A = 160
P_DIR_V1B = 144
P_DIR_V2A = 159
P_DIR_V2B = 145

P_ESQ_H1 = 362
P_ESQ_H2 = 263
P_ESQ_V1A = 385
P_ESQ_V1B = 380
P_ESQ_V2A = 386
P_ESQ_V2B = 374


# APARÊNCIA DOS GRÁFICOS

CORES_MODELOS = [
    "red",
    "mediumseagreen",
    "blue",
    "orange",
    "purple",
]

MARCADORES_MODELOS = [
    "o",
    "s",
    "^",
    "d",
    "*",
]

TAMANHOS_MODELOS = [
    190,
    190,
    210,
    210,
    280,
]


clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)


def validar_ambiente():
    print("\nVerificando ambiente...")

    versao_python = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )

    print(f"Python: {versao_python}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"MediaPipe: {mp.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"PyTorch: {torch.__version__}")

    if sys.version_info[:2] != (3, 12):
        print(
            "Aviso: o ambiente recomendado para este projeto é Python 3.12."
        )

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Esta instalação do MediaPipe não possui a API mp.solutions "
            "utilizada pelo projeto. Execute: "
            "python -m pip install -r requirements.txt"
        )

    if mp.__version__ != "0.10.21":
        print(
            "Aviso: o projeto foi preparado para MediaPipe 0.10.21."
        )

    if np.__version__ != "1.26.4":
        print(
            "Aviso: o ambiente recomendado utiliza NumPy 1.26.4."
        )

    print("Ambiente verificado com sucesso.")


def nome_modelo(modelo_arquivo):
    return Path(modelo_arquivo).stem.replace(
        "yolov8",
        "YOLOv8",
    )


def letra_modelo(modelo_nome):
    return modelo_nome.replace("YOLOv8", "")


def carregar_modelo(modelo_arquivo):
    caminho_modelo = BASE_DIR / modelo_arquivo

    try:
        if caminho_modelo.exists():
            return YOLO(str(caminho_modelo))

        return YOLO(modelo_arquivo)

    except Exception as erro:
        raise RuntimeError(
            f"Não foi possível carregar {modelo_arquivo}. "
            "Verifique se o peso está na pasta avaliar_modelos "
            "ou se há conexão com a internet para o download automático."
        ) from erro


def obter_hardware():
    if DEVICE == "cpu":
        return "CPU"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE está configurado para GPU, mas CUDA não foi encontrada. "
            "Verifique o driver NVIDIA e a instalação do PyTorch com CUDA."
        )

    return "GPU"


def sincronizar_gpu():
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def aplicar_gamma(img_bgr, gamma):
    if gamma <= 0:
        return img_bgr

    inv_gamma = 1.0 / gamma

    tabela = np.array(
        [
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8,
    )

    return cv2.LUT(
        img_bgr,
        tabela,
    )


def aplicar_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2LAB,
    )

    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )


def preprocessar_roi_face(img_bgr):
    saida = img_bgr.copy()

    if ATIVAR_GAMMA_ADAPTIVO:
        gray = cv2.cvtColor(
            saida,
            cv2.COLOR_BGR2GRAY,
        )

        media = gray.mean()

        if media < BRILHO_REFERENCIA:
            intensidade = (
                BRILHO_REFERENCIA - media
            ) / BRILHO_REFERENCIA

            gamma = (
                GAMMA_MIN
                + (GAMMA_MAX - GAMMA_MIN) * intensidade
            )

            gamma = max(
                GAMMA_MIN,
                min(GAMMA_MAX, gamma),
            )

            saida = aplicar_gamma(
                saida,
                gamma,
            )

    if ATIVAR_CLAHE:
        saida = aplicar_clahe_bgr(
            saida
        )

    return saida


def ponto_px(
    face,
    idx,
    largura,
    altura,
):
    return (
        int(face.landmark[idx].x * largura),
        int(face.landmark[idx].y * altura),
    )


def distancia(
    p1,
    p2,
):
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1],
    )


def calcular_ear_olho(
    face,
    largura,
    altura,
    h1,
    h2,
    v1a,
    v1b,
    v2a,
    v2b,
):
    ph1 = ponto_px(
        face,
        h1,
        largura,
        altura,
    )

    ph2 = ponto_px(
        face,
        h2,
        largura,
        altura,
    )

    pv1a = ponto_px(
        face,
        v1a,
        largura,
        altura,
    )

    pv1b = ponto_px(
        face,
        v1b,
        largura,
        altura,
    )

    pv2a = ponto_px(
        face,
        v2a,
        largura,
        altura,
    )

    pv2b = ponto_px(
        face,
        v2b,
        largura,
        altura,
    )

    horizontal = distancia(
        ph1,
        ph2,
    )

    vertical1 = distancia(
        pv1a,
        pv1b,
    )

    vertical2 = distancia(
        pv2a,
        pv2b,
    )

    if horizontal == 0:
        return 0.0

    return (
        vertical1 + vertical2
    ) / (
        2.0 * horizontal
    )


def suavizar_ear(
    valor_atual,
    valor_anterior,
):
    if valor_anterior is None:
        return valor_atual

    return (
        ALPHA_EAR * valor_atual
        + (1.0 - ALPHA_EAR) * valor_anterior
    )


class AnalisadorFacial:
    def __init__(self):
        self.face_mesh = (
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        )

        self.face_detection = (
            mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.45,
            )
        )

    def fechar(self):
        self.face_mesh.close()
        self.face_detection.close()

    def detectar_face_roi_superior(
        self,
        img_bgr,
    ):
        h, w, _ = img_bgr.shape

        img_rgb = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB,
        )

        det = self.face_detection.process(
            img_rgb
        )

        if not det or not det.detections:
            return None

        caixa = (
            det.detections[0]
            .location_data
            .relative_bounding_box
        )

        x = int(caixa.xmin * w)
        y = int(caixa.ymin * h)
        bw = int(caixa.width * w)
        bh = int(caixa.height * h)

        if (
            bw < MIN_LARGURA_FACE
            or bh < MIN_ALTURA_FACE
        ):
            return None

        x1 = int(
            max(
                0,
                x - bw * FATOR_EXPANSAO_LATERAL,
            )
        )

        x2 = int(
            min(
                w,
                x + bw + bw * FATOR_EXPANSAO_LATERAL,
            )
        )

        y1 = int(
            max(
                0,
                y - bh * FATOR_EXPANSAO_SUPERIOR,
            )
        )

        y2 = int(
            min(
                h,
                y + bh * (
                    0.50
                    + FATOR_EXPANSAO_INFERIOR
                ),
            )
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None

        return (
            x1,
            y1,
            x2,
            y2,
        )

    def extrair_ear(
        self,
        frame_bgr,
    ):
        roi = self.detectar_face_roi_superior(
            frame_bgr
        )

        if roi is None:
            return None, None

        x1, y1, x2, y2 = roi

        face_roi = frame_bgr[
            y1:y2,
            x1:x2,
        ]

        if face_roi.size == 0:
            return None, None

        largura_face = x2 - x1
        altura_face = y2 - y1

        face_roi = preprocessar_roi_face(
            face_roi
        )

        face_zoom = cv2.resize(
            face_roi,
            (
                int(
                    largura_face
                    * FATOR_ZOOM_FACE
                ),
                int(
                    altura_face
                    * FATOR_ZOOM_FACE
                ),
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        face_zoom_rgb = cv2.cvtColor(
            face_zoom,
            cv2.COLOR_BGR2RGB,
        )

        resultado = self.face_mesh.process(
            face_zoom_rgb
        )

        if (
            not resultado
            or not resultado.multi_face_landmarks
        ):
            return None, None

        face = (
            resultado.multi_face_landmarks[0]
        )

        altura_zoom, largura_zoom, _ = (
            face_zoom.shape
        )

        ear_dir = calcular_ear_olho(
            face,
            largura_zoom,
            altura_zoom,
            P_DIR_H1,
            P_DIR_H2,
            P_DIR_V1A,
            P_DIR_V1B,
            P_DIR_V2A,
            P_DIR_V2B,
        )

        ear_esq = calcular_ear_olho(
            face,
            largura_zoom,
            altura_zoom,
            P_ESQ_H1,
            P_ESQ_H2,
            P_ESQ_V1A,
            P_ESQ_V1B,
            P_ESQ_V2A,
            P_ESQ_V2B,
        )

        return (
            ear_dir,
            ear_esq,
        )


def listar_videos_dataset():
    if not PASTA_DATASET.exists():
        raise FileNotFoundError(
            f"Pasta dataset não encontrada: {PASTA_DATASET}"
        )

    videos = []

    print("\nVerificando dataset...\n")

    for classe in CLASSES:
        pasta_classe = (
            PASTA_DATASET / classe
        )

        if not pasta_classe.exists():
            raise FileNotFoundError(
                f"Pasta da classe não encontrada: {pasta_classe}"
            )

        arquivos = sorted(
            [
                arquivo
                for arquivo
                in pasta_classe.iterdir()
                if (
                    arquivo.is_file()
                    and arquivo.suffix.lower()
                    in EXTENSOES_VIDEO
                )
            ],
            key=lambda p: p.name.lower(),
        )

        if not arquivos:
            raise RuntimeError(
                f"Nenhum vídeo encontrado em: {pasta_classe}"
            )

        print(
            f"{NOMES_EXIBICAO[classe]:20s} "
            f"{len(arquivos):3d} vídeos"
        )

        if len(arquivos) != 30:
            print(
                f"Aviso: a classe {classe} possui "
                f"{len(arquivos)} vídeos; eram esperados 30."
            )

        for arquivo in arquivos:
            videos.append(
                {
                    "arquivo": arquivo,
                    "classe_real": classe,
                }
            )

    print(
        f"\nTotal: {len(videos)} vídeos"
    )

    if len(videos) != 120:
        print(
            "Aviso: o total encontrado é diferente de 120 vídeos. "
            "A avaliação continuará usando todos os vídeos encontrados."
        )

    return videos


def calibrar_ear_video(
    arquivo_video,
):
    captura = cv2.VideoCapture(
        str(arquivo_video)
    )

    if not captura.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir o vídeo para calibração: "
            f"{arquivo_video}"
        )

    fps_video = captura.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not fps_video
        or fps_video <= 0
    ):
        fps_video = 30.0

    analisador = AnalisadorFacial()

    amostras_dir = []
    amostras_esq = []

    ear_filtrado_dir = None
    ear_filtrado_esq = None

    indice_frame = 0

    try:
        while True:
            ret, frame = captura.read()

            if not ret:
                break

            timestamp_video = (
                indice_frame / fps_video
            )

            if (
                timestamp_video
                > TEMPO_CALIBRACAO
            ):
                break

            frame = cv2.resize(
                frame,
                RESOLUCAO_CAPTURA,
            )

            ear_dir, ear_esq = (
                analisador.extrair_ear(
                    frame
                )
            )

            if (
                ear_dir is not None
                and ear_esq is not None
            ):
                ear_filtrado_dir = (
                    suavizar_ear(
                        ear_dir,
                        ear_filtrado_dir,
                    )
                )

                ear_filtrado_esq = (
                    suavizar_ear(
                        ear_esq,
                        ear_filtrado_esq,
                    )
                )

                amostras_dir.append(
                    ear_filtrado_dir
                )

                amostras_esq.append(
                    ear_filtrado_esq
                )

            indice_frame += 1

    finally:
        captura.release()
        analisador.fechar()

    if (
        not amostras_dir
        or not amostras_esq
    ):
        return {
            "ear_base_dir": EAR_BASE_FALLBACK,
            "ear_base_esq": EAR_BASE_FALLBACK,
            "ear_limiar_dir": EAR_LIMIAR_FALLBACK,
            "ear_limiar_esq": EAR_LIMIAR_FALLBACK,
            "fallback": True,
        }

    ear_base_dir = (
        sum(amostras_dir)
        / len(amostras_dir)
    )

    ear_base_esq = (
        sum(amostras_esq)
        / len(amostras_esq)
    )

    ear_limiar_dir = max(
        LIMIAR_EAR_MIN,
        min(
            LIMIAR_EAR_MAX,
            ear_base_dir
            * FATOR_LIMIAR_EAR,
        ),
    )

    ear_limiar_esq = max(
        LIMIAR_EAR_MIN,
        min(
            LIMIAR_EAR_MAX,
            ear_base_esq
            * FATOR_LIMIAR_EAR,
        ),
    )

    return {
        "ear_base_dir": ear_base_dir,
        "ear_base_esq": ear_base_esq,
        "ear_limiar_dir": ear_limiar_dir,
        "ear_limiar_esq": ear_limiar_esq,
        "fallback": False,
    }


def calibrar_todos_videos(
    videos,
):
    calibracoes = {}

    print()
    print(
        "Calibração EAR individual por vídeo"
    )

    total = len(videos)
    total_fallback = 0

    for indice, item in enumerate(
        videos,
        start=1,
    ):
        arquivo = item["arquivo"]

        calibracao = calibrar_ear_video(
            arquivo
        )

        calibracoes[str(arquivo)] = (
            calibracao
        )

        if calibracao["fallback"]:
            total_fallback += 1

        caminho_relativo = (
            arquivo.relative_to(
                PASTA_DATASET
            )
        )

        status = (
            "fallback"
            if calibracao["fallback"]
            else "ok"
        )

        print(
            f"[{indice:03d}/{total:03d}] "
            f"{caminho_relativo} | {status}"
        )

    print()

    if total_fallback > 0:
        print(
            f"Aviso: {total_fallback} vídeo(s) não forneceram "
            "amostras EAR válidas na calibração e utilizaram "
            "os valores de segurança."
        )

    print(
        "Calibração individual concluída."
    )

    return calibracoes


def detectar_celular(
    modelo,
    img_bgr,
):
    resultado = modelo.predict(
        source=img_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONFIANCA_CELULAR,
        classes=[67],
        verbose=False,
        device=DEVICE,
    )

    melhor_conf = 0.0

    for objetos in resultado:
        if (
            objetos.boxes is None
            or len(objetos.boxes) == 0
        ):
            continue

        for dados in objetos.boxes:
            conf = float(
                dados.conf[0]
            )

            if conf > melhor_conf:
                melhor_conf = conf

    return (
        melhor_conf > 0.0,
        melhor_conf,
    )


def finalizar_fechamento(
    inicio_fechado,
    timestamp_atual,
    maior_tempo_fechado,
    evento_piscada,
    evento_sonolencia,
):
    if inicio_fechado is None:
        return (
            maior_tempo_fechado,
            evento_piscada,
            evento_sonolencia,
        )

    duracao = max(
        0.0,
        timestamp_atual
        - inicio_fechado,
    )

    maior_tempo_fechado = max(
        maior_tempo_fechado,
        duracao,
    )

    # A piscada prolongada só é confirmada quando os olhos reabrem.
    # Se o fechamento atingir 2 s antes da reabertura, é sonolência.
    if duracao >= TEMPO_FECHADO:
        evento_sonolencia = True

    elif (
        duracao
        >= TEMPO_PISCADA_PROLONGADA
    ):
        evento_piscada = True

    return (
        maior_tempo_fechado,
        evento_piscada,
        evento_sonolencia,
    )


def escolher_evento_detectado(
    evento_celular,
    evento_sonolencia,
    evento_piscada,
):
    # Esta prioridade só é utilizada quando a classe-alvo
    # do vídeo não foi detectada ou quando a classe real é Normal.
    if evento_celular:
        return "Det_celular"

    if evento_sonolencia:
        return "Sonolencia"

    if evento_piscada:
        return "Pisc_prolongada"

    return "Normal"


def classificar_video(
    classe_real,
    evento_celular,
    evento_sonolencia,
    evento_piscada,
):
    # A pasta informa a classe-alvo do vídeo.
    # Se o evento correspondente à classe-alvo foi detectado,
    # outras distrações detectadas no mesmo vídeo não anulam o acerto.

    if classe_real == "Normal":
        return escolher_evento_detectado(
            evento_celular,
            evento_sonolencia,
            evento_piscada,
        )

    if (
        classe_real == "Det_celular"
        and evento_celular
    ):
        return "Det_celular"

    if (
        classe_real == "Sonolencia"
        and evento_sonolencia
    ):
        return "Sonolencia"

    if (
        classe_real
        == "Pisc_prolongada"
        and evento_piscada
    ):
        return "Pisc_prolongada"

    return escolher_evento_detectado(
        evento_celular,
        evento_sonolencia,
        evento_piscada,
    )


def avaliar_video(
    modelo,
    arquivo_video,
    classe_real,
    calibracao,
):
    captura = cv2.VideoCapture(
        str(arquivo_video)
    )

    if not captura.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir o vídeo: "
            f"{arquivo_video}"
        )

    fps_original = captura.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not fps_original
        or fps_original <= 0
    ):
        fps_original = 30.0

    total_frames_arquivo = int(
        captura.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duracao_video = (
        total_frames_arquivo
        / fps_original
        if total_frames_arquivo > 0
        else 0.0
    )

    analisador = AnalisadorFacial()

    ear_filtrado_dir = None
    ear_filtrado_esq = None

    contador_fechado = 0
    inicio_fechado = None

    evento_celular = False
    evento_piscada = False
    evento_sonolencia = False

    maior_tempo_fechado = 0.0
    maior_conf_celular = 0.0
    quantidade_deteccoes_celular = 0

    frames_processados = 0
    frames_face_valida = 0

    tempo_processamento = 0.0

    ultimo_timestamp = 0.0
    indice_frame = 0

    try:
        while True:
            ret, frame_original = (
                captura.read()
            )

            if not ret:
                break

            timestamp_video = (
                indice_frame
                / fps_original
            )

            ultimo_timestamp = (
                timestamp_video
            )

            sincronizar_gpu()

            inicio_medicao = (
                time.perf_counter()
            )

            frame_captura = cv2.resize(
                frame_original,
                RESOLUCAO_CAPTURA,
            )

            ear_dir, ear_esq = (
                analisador.extrair_ear(
                    frame_captura
                )
            )

            if (
                ear_dir is None
                or ear_esq is None
            ):
                # Se a face deixa de ser localizada, o fechamento em curso
                # é encerrado sem criar artificialmente um novo evento.
                contador_fechado = 0
                inicio_fechado = None

            else:
                frames_face_valida += 1

                ear_filtrado_dir = (
                    suavizar_ear(
                        ear_dir,
                        ear_filtrado_dir,
                    )
                )

                ear_filtrado_esq = (
                    suavizar_ear(
                        ear_esq,
                        ear_filtrado_esq,
                    )
                )

                olho_dir_fechado = (
                    ear_filtrado_dir
                    <= calibracao[
                        "ear_limiar_dir"
                    ]
                )

                olho_esq_fechado = (
                    ear_filtrado_esq
                    <= calibracao[
                        "ear_limiar_esq"
                    ]
                )

                olhos_fechados = (
                    olho_dir_fechado
                    and olho_esq_fechado
                )

                if olhos_fechados:
                    contador_fechado += 1

                    if (
                        contador_fechado
                        >= FRAMES_CONSECUTIVOS_FECHADO
                        and inicio_fechado
                        is None
                    ):
                        inicio_fechado = (
                            timestamp_video
                        )

                    if (
                        inicio_fechado
                        is not None
                    ):
                        duracao_atual = max(
                            0.0,
                            timestamp_video
                            - inicio_fechado,
                        )

                        maior_tempo_fechado = max(
                            maior_tempo_fechado,
                            duracao_atual,
                        )

                        # Enquanto os olhos permanecem fechados,
                        # não se confirma piscada prolongada.
                        # O sistema aguarda a reabertura ou 2 s contínuos.
                        if (
                            duracao_atual
                            >= TEMPO_FECHADO
                        ):
                            evento_sonolencia = (
                                True
                            )

                else:
                    if (
                        inicio_fechado
                        is not None
                    ):
                        (
                            maior_tempo_fechado,
                            evento_piscada,
                            evento_sonolencia,
                        ) = finalizar_fechamento(
                            inicio_fechado,
                            timestamp_video,
                            maior_tempo_fechado,
                            evento_piscada,
                            evento_sonolencia,
                        )

                    contador_fechado = 0
                    inicio_fechado = None

            if (
                indice_frame
                % PULAR_FRAMES_YOLO
                == 0
            ):
                frame_yolo = cv2.resize(
                    frame_captura,
                    RESOLUCAO_YOLO,
                )

                detectado, conf = (
                    detectar_celular(
                        modelo,
                        frame_yolo,
                    )
                )

                if detectado:
                    quantidade_deteccoes_celular += 1

                    maior_conf_celular = max(
                        maior_conf_celular,
                        conf,
                    )

                    if (
                        quantidade_deteccoes_celular
                        >= MIN_DETECCOES_CELULAR
                    ):
                        evento_celular = True

            sincronizar_gpu()

            fim_medicao = (
                time.perf_counter()
            )

            tempo_processamento += (
                fim_medicao
                - inicio_medicao
            )

            frames_processados += 1
            indice_frame += 1

    finally:
        captura.release()
        analisador.fechar()

    if inicio_fechado is not None:
        timestamp_final = (
            ultimo_timestamp
            + (1.0 / fps_original)
        )

        (
            maior_tempo_fechado,
            evento_piscada,
            evento_sonolencia,
        ) = finalizar_fechamento(
            inicio_fechado,
            timestamp_final,
            maior_tempo_fechado,
            evento_piscada,
            evento_sonolencia,
        )

    classe_prevista = classificar_video(
        classe_real,
        evento_celular,
        evento_sonolencia,
        evento_piscada,
    )

    fps_processamento = (
        frames_processados
        / tempo_processamento
        if tempo_processamento > 0
        else 0.0
    )

    percentual_face_valida = (
        frames_face_valida
        / frames_processados
        if frames_processados > 0
        else 0.0
    )

    return {
        "classe_prevista": classe_prevista,
        "frames": frames_processados,
        "fps_original": fps_original,
        "duracao_video": duracao_video,
        "tempo_processamento": tempo_processamento,
        "fps_processamento": fps_processamento,
        "evento_celular": int(
            evento_celular
        ),
        "deteccoes_celular": (
            quantidade_deteccoes_celular
        ),
        "maior_conf_celular": (
            maior_conf_celular
        ),
        "evento_piscada": int(
            evento_piscada
        ),
        "evento_sonolencia": int(
            evento_sonolencia
        ),
        "maior_tempo_olhos_fechados": (
            maior_tempo_fechado
        ),
        "percentual_face_valida": (
            percentual_face_valida
        ),
    }


def obter_frame_aquecimento(
    videos,
):
    captura = cv2.VideoCapture(
        str(videos[0]["arquivo"])
    )

    if not captura.isOpened():
        raise RuntimeError(
            "Não foi possível abrir o primeiro "
            "vídeo para aquecimento."
        )

    ret, frame = captura.read()

    captura.release()

    if not ret:
        raise RuntimeError(
            "Não foi possível ler frame "
            "para aquecimento."
        )

    frame = cv2.resize(
        frame,
        RESOLUCAO_CAPTURA,
    )

    return cv2.resize(
        frame,
        RESOLUCAO_YOLO,
    )


def aquecer_modelo(
    modelo,
    frame,
):
    _ = modelo.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=CONFIANCA_CELULAR,
        classes=[67],
        verbose=False,
        device=DEVICE,
    )

    sincronizar_gpu()


def calcular_metricas(
    classes_reais,
    classes_previstas,
):
    acuracia = accuracy_score(
        classes_reais,
        classes_previstas,
    )

    (
        precisao_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        classes_reais,
        classes_previstas,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    (
        precisao_classe,
        recall_classe,
        f1_classe,
        suporte,
    ) = precision_recall_fscore_support(
        classes_reais,
        classes_previstas,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    por_classe = []

    for i, classe in enumerate(
        CLASSES
    ):
        por_classe.append(
            {
                "classe": classe,
                "precisao": float(
                    precisao_classe[i]
                ),
                "recall": float(
                    recall_classe[i]
                ),
                "f1_score": float(
                    f1_classe[i]
                ),
                "suporte": int(
                    suporte[i]
                ),
            }
        )

    return {
        "acuracia": float(
            acuracia
        ),
        "precisao": float(
            precisao_macro
        ),
        "recall": float(
            recall_macro
        ),
        "f1_score": float(
            f1_macro
        ),
        "por_classe": por_classe,
    }


def gerar_matriz_confusao(
    classes_reais,
    classes_previstas,
    arquivo_imagem,
    arquivo_contagens,
):
    # sklearn gera linhas=True e colunas=Predicted.
    # O trabalho utiliza linhas=Predicted e colunas=True.
    matriz_true_pred = confusion_matrix(
        classes_reais,
        classes_previstas,
        labels=CLASSES,
    )

    matriz_contagens = (
        matriz_true_pred.T
    )

    somas_colunas = (
        matriz_contagens.sum(
            axis=0,
            keepdims=True,
        )
    )

    matriz_normalizada = np.divide(
        matriz_contagens.astype(float),
        somas_colunas,
        out=np.zeros_like(
            matriz_contagens,
            dtype=float,
        ),
        where=somas_colunas != 0,
    )

    with arquivo_contagens.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Predicted / True"
            ]
            + [
                NOMES_EXIBICAO[c]
                for c in CLASSES
            ]
        )

        for i, classe in enumerate(
            CLASSES
        ):
            writer.writerow(
                [
                    NOMES_EXIBICAO[
                        classe
                    ]
                ]
                + matriz_contagens[
                    i
                ].tolist()
            )

    quantidade_classes = len(
        CLASSES
    )

    fig, ax = plt.subplots(
        figsize=(10, 9),
        facecolor="white",
    )

    ax.set_facecolor(
        "white"
    )

    imagem = ax.imshow(
        matriz_normalizada,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )

    posicoes = np.arange(
        quantidade_classes
    )

    ax.set_xticks(
        posicoes
    )

    ax.set_yticks(
        posicoes
    )

    ax.set_xticklabels(
        [
            NOMES_EXIBICAO[c]
            for c in CLASSES
        ],
        fontsize=16,
    )

    ax.set_yticklabels(
        [
            NOMES_EXIBICAO[c]
            for c in CLASSES
        ],
        fontsize=16,
        rotation=90,
        va="center",
    )

    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        direction="out",
        length=3,
        width=1.1,
        pad=7,
    )

    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        right=False,
        direction="out",
        length=3,
        width=1.1,
        pad=7,
    )

    ax.set_xlabel(
        "True",
        fontsize=16,
        labelpad=10,
    )

    ax.set_ylabel(
        "Predicted",
        fontsize=16,
        labelpad=12,
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=15,
        pad=18,
    )

    for linha in range(
        quantidade_classes
    ):
        for coluna in range(
            quantidade_classes
        ):
            valor = (
                matriz_normalizada[
                    linha,
                    coluna,
                ]
            )

            cor_texto = (
                "white"
                if valor >= 0.50
                else "black"
            )

            ax.text(
                coluna,
                linha,
                f"{valor:.2f}",
                ha="center",
                va="center",
                fontsize=16,
                color=cor_texto,
            )

    ax.set_xlim(
        -0.5,
        quantidade_classes - 0.5,
    )

    ax.set_ylim(
        quantidade_classes - 0.5,
        -0.5,
    )

    for borda in ax.spines.values():
        borda.set_visible(False)

    ax.grid(False)

    barra_cores = fig.colorbar(
        imagem,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    barra_cores.set_ticks(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
        ]
    )

    barra_cores.set_ticklabels(
        [
            "0.0",
            "0.2",
            "0.4",
            "0.6",
            "0.8",
        ]
    )

    barra_cores.ax.tick_params(
        labelsize=12,
        direction="out",
        length=3,
        width=0.8,
    )

    barra_cores.outline.set_visible(
        False
    )

    for borda in barra_cores.ax.spines.values():
        borda.set_visible(False)

    plt.tight_layout()

    plt.savefig(
        arquivo_imagem,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    plt.close(fig)


def salvar_resultados_videos(
    resultados_videos,
    arquivo_saida,
):
    campos = [
        "arquivo",
        "classe_real",
        "classe_prevista",
        "resultado",
        "frames",
        "fps_original",
        "duracao_video",
        "tempo_processamento",
        "fps_processamento",
        "evento_celular",
        "deteccoes_celular",
        "maior_conf_celular",
        "evento_piscada",
        "evento_sonolencia",
        "maior_tempo_olhos_fechados",
        "percentual_face_valida",
    ]

    with arquivo_saida.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=campos,
        )

        writer.writeheader()

        writer.writerows(
            resultados_videos
        )


def salvar_metricas(
    metricas,
    fps_global,
    pasta_modelo,
):
    with (
        pasta_modelo
        / "metricas.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Acuracia",
                "Precisao",
                "Recall",
                "F1-score",
                "FPS",
            ]
        )

        writer.writerow(
            [
                f"{metricas['acuracia']:.6f}",
                f"{metricas['precisao']:.6f}",
                f"{metricas['recall']:.6f}",
                f"{metricas['f1_score']:.6f}",
                f"{fps_global:.6f}",
            ]
        )

    with (
        pasta_modelo
        / "metricas_por_classe.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "classe",
                "precisao",
                "recall",
                "f1_score",
                "suporte",
            ],
        )

        writer.writeheader()

        for item in metricas[
            "por_classe"
        ]:
            writer.writerow(
                {
                    "classe": (
                        NOMES_EXIBICAO[
                            item["classe"]
                        ]
                    ),
                    "precisao": (
                        f"{item['precisao']:.6f}"
                    ),
                    "recall": (
                        f"{item['recall']:.6f}"
                    ),
                    "f1_score": (
                        f"{item['f1_score']:.6f}"
                    ),
                    "suporte": (
                        item["suporte"]
                    ),
                }
            )


def gerar_grafico_acuracia_fps(
    resumo_modelos,
    hardware,
    arquivo_saida,
):
    modelos = [
        item["modelo"]
        for item in resumo_modelos
    ]

    fps = [
        item["fps"]
        for item in resumo_modelos
    ]

    acuracia = [
        item["acuracia"]
        for item in resumo_modelos
    ]

    plt.figure(
        figsize=(11, 7)
    )

    for i in range(
        len(modelos)
    ):
        plt.scatter(
            fps[i],
            acuracia[i],
            color=CORES_MODELOS[i],
            marker=MARCADORES_MODELOS[i],
            s=TAMANHOS_MODELOS[i],
            edgecolor="black",
            linewidth=1.5,
            label=(
                f"{modelos[i]} "
                f"+ MediaPipe"
            ),
            zorder=3,
        )

    plt.title(
        hardware,
        fontsize=20,
        fontweight="bold",
        pad=15,
    )

    plt.xlabel(
        "FPS",
        fontsize=18,
    )

    plt.ylabel(
        "Acurácia",
        fontsize=18,
    )

    menor_acuracia = min(
        acuracia
    )

    limite_y_min = (
        0.86
        if menor_acuracia >= 0.86
        else max(
            0.0,
            menor_acuracia - 0.03,
        )
    )

    plt.ylim(
        limite_y_min,
        1.00,
    )

    menor_fps = min(
        fps
    )

    maior_fps = max(
        fps
    )

    intervalo_fps = (
        maior_fps - menor_fps
    )

    margem_x = max(
        0.5,
        intervalo_fps * 0.08,
    )

    limite_x_min = max(
        0.0,
        menor_fps - margem_x,
    )

    limite_x_max = (
        maior_fps + margem_x
    )

    if (
        limite_x_max
        <= limite_x_min
    ):
        limite_x_max = (
            limite_x_min + 1.0
        )

    plt.xlim(
        limite_x_min,
        limite_x_max,
    )

    plt.xticks(
        fontsize=13
    )

    plt.yticks(
        fontsize=13
    )

    plt.grid(
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )

    plt.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        arquivo_saida,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close()


def salvar_resumo_geral(
    resumo_modelos,
    arquivo_saida,
):
    with arquivo_saida.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Modelo",
                "Acuracia",
                "Precisao",
                "Recall",
                "F1-score",
                "FPS",
            ]
        )

        for item in resumo_modelos:
            writer.writerow(
                [
                    (
                        f"{item['modelo']} "
                        f"+ MediaPipe"
                    ),
                    f"{item['acuracia']:.6f}",
                    f"{item['precisao']:.6f}",
                    f"{item['recall']:.6f}",
                    f"{item['f1_score']:.6f}",
                    f"{item['fps']:.6f}",
                ]
            )


def gerar_tabela_metricas_png(
    resumo_modelos,
    arquivo_saida,
):
    colunas = [
        "Modelos",
        "Acurácia",
        "Precisão",
        "Recall",
        "F1-score",
        "FPS",
    ]

    dados = []

    for item in resumo_modelos:
        dados.append(
            [
                letra_modelo(
                    item["modelo"]
                ),
                f"{item['acuracia']:.3f}",
                f"{item['precisao']:.3f}",
                f"{item['recall']:.3f}",
                f"{item['f1_score']:.3f}",
                f"{item['fps']:.1f}",
            ]
        )

    fig, ax = plt.subplots(
        figsize=(8.6, 3.8),
        facecolor="white",
    )

    ax.axis(
        "off"
    )

    tabela = ax.table(
        cellText=dados,
        colLabels=colunas,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[
            0.02,
            0.04,
            0.96,
            0.92,
        ],
    )

    tabela.auto_set_font_size(
        False
    )

    tabela.set_fontsize(
        12
    )

    tabela.scale(
        1.0,
        1.45,
    )

    total_linhas = (
        len(dados) + 1
    )

    total_colunas = len(
        colunas
    )

    for (
        linha,
        coluna,
    ), celula in tabela.get_celld().items():
        celula.set_facecolor(
            "white"
        )

        celula.set_edgecolor(
            "black"
        )

        celula.set_linewidth(
            0.0
        )

        if linha == 0:
            celula.get_text().set_weight(
                "bold"
            )

            celula.visible_edges = (
                "TB"
            )

            celula.set_linewidth(
                0.8
            )

        elif linha == (
            total_linhas - 1
        ):
            celula.visible_edges = (
                "B"
            )

            celula.set_linewidth(
                0.8
            )

        else:
            celula.visible_edges = (
                ""
            )

        if (
            coluna == 0
            and linha > 0
        ):
            celula.get_text().set_style(
                "italic"
            )

    plt.savefig(
        arquivo_saida,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    plt.close(fig)


def salvar_ambiente(
    hardware,
    pasta_hardware,
):
    arquivo = (
        pasta_hardware
        / f"ambiente_{hardware}.txt"
    )

    with arquivo.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Hardware selecionado: "
            f"{hardware}\n"
        )

        f.write(
            f"DEVICE: {DEVICE}\n"
        )

        if hardware == "GPU":
            f.write(
                "GPU: "
                f"{torch.cuda.get_device_name(0)}\n"
            )

            f.write(
                "CUDA PyTorch: "
                f"{torch.version.cuda}\n"
            )

        f.write(
            f"Sistema: "
            f"{platform.platform()}\n"
        )

        f.write(
            f"Python: "
            f"{sys.version.split()[0]}\n"
        )

        f.write(
            f"OpenCV: "
            f"{cv2.__version__}\n"
        )

        f.write(
            f"MediaPipe: "
            f"{mp.__version__}\n"
        )

        f.write(
            f"NumPy: "
            f"{np.__version__}\n"
        )

        f.write(
            f"Ultralytics: "
            f"{ultralytics.__version__}\n"
        )

        f.write(
            f"scikit-learn: "
            f"{sklearn.__version__}\n"
        )

        f.write(
            f"PyTorch: "
            f"{torch.__version__}\n"
        )


def liberar_modelo(
    modelo,
):
    del modelo

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    validar_ambiente()

    hardware = obter_hardware()

    print()
    print(
        "AVALIAÇÃO YOLOv8 + MediaPipe"
    )

    print(
        f"Dispositivo: {hardware}"
    )

    print(
        f"Dataset: {PASTA_DATASET}"
    )

    videos = listar_videos_dataset()

    pasta_hardware = (
        PASTA_RESULTADOS
        / hardware
    )

    pasta_hardware.mkdir(
        parents=True,
        exist_ok=True,
    )

    salvar_ambiente(
        hardware,
        pasta_hardware,
    )

    # A calibração é individual por vídeo e é feita uma única vez.
    # A mesma referência EAR de cada vídeo é usada nas cinco variantes,
    # evitando diferenças de calibração entre os modelos.
    calibracoes = calibrar_todos_videos(
        videos
    )

    frame_aquecimento = (
        obter_frame_aquecimento(
            videos
        )
    )

    resumo_modelos = []

    for (
        indice_modelo,
        modelo_arquivo,
    ) in enumerate(
        MODELOS,
        start=1,
    ):
        modelo_nome = nome_modelo(
            modelo_arquivo
        )

        print()
        print(
            f"[{indice_modelo}/{len(MODELOS)}] "
            f"{modelo_nome} + MediaPipe - "
            f"{hardware}"
        )

        pasta_modelo = (
            pasta_hardware
            / modelo_nome
        )

        pasta_modelo.mkdir(
            parents=True,
            exist_ok=True,
        )

        modelo = carregar_modelo(
            modelo_arquivo
        )

        print(
            "Aquecendo modelo..."
        )

        aquecer_modelo(
            modelo,
            frame_aquecimento,
        )

        resultados_videos = []

        classes_reais = []
        classes_previstas = []

        total_frames_modelo = 0
        tempo_total_modelo = 0.0

        total_videos = len(
            videos
        )

        for indice_video, item in enumerate(
            videos,
            start=1,
        ):
            arquivo = item[
                "arquivo"
            ]

            classe_real = item[
                "classe_real"
            ]

            calibracao = calibracoes[
                str(arquivo)
            ]

            resultado_video = (
                avaliar_video(
                    modelo,
                    arquivo,
                    classe_real,
                    calibracao,
                )
            )

            classe_prevista = (
                resultado_video[
                    "classe_prevista"
                ]
            )

            acertou = (
                classe_real
                == classe_prevista
            )

            classes_reais.append(
                classe_real
            )

            classes_previstas.append(
                classe_prevista
            )

            total_frames_modelo += (
                resultado_video[
                    "frames"
                ]
            )

            tempo_total_modelo += (
                resultado_video[
                    "tempo_processamento"
                ]
            )

            caminho_relativo = (
                arquivo.relative_to(
                    PASTA_DATASET
                )
            )

            linha = {
                "arquivo": str(
                    caminho_relativo
                ),
                "classe_real": (
                    classe_real
                ),
                "classe_prevista": (
                    classe_prevista
                ),
                "resultado": (
                    "ACERTO"
                    if acertou
                    else "ERRO"
                ),
                "frames": (
                    resultado_video[
                        "frames"
                    ]
                ),
                "fps_original": (
                    f"{resultado_video['fps_original']:.6f}"
                ),
                "duracao_video": (
                    f"{resultado_video['duracao_video']:.6f}"
                ),
                "tempo_processamento": (
                    f"{resultado_video['tempo_processamento']:.6f}"
                ),
                "fps_processamento": (
                    f"{resultado_video['fps_processamento']:.6f}"
                ),
                "evento_celular": (
                    resultado_video[
                        "evento_celular"
                    ]
                ),
                "deteccoes_celular": (
                    resultado_video[
                        "deteccoes_celular"
                    ]
                ),
                "maior_conf_celular": (
                    f"{resultado_video['maior_conf_celular']:.6f}"
                ),
                "evento_piscada": (
                    resultado_video[
                        "evento_piscada"
                    ]
                ),
                "evento_sonolencia": (
                    resultado_video[
                        "evento_sonolencia"
                    ]
                ),
                "maior_tempo_olhos_fechados": (
                    f"{resultado_video['maior_tempo_olhos_fechados']:.6f}"
                ),
                "percentual_face_valida": (
                    f"{resultado_video['percentual_face_valida']:.6f}"
                ),
            }

            resultados_videos.append(
                linha
            )

            # ACERTO/ERRO permanece salvo no CSV,
            # mas não é exibido no terminal.
            print(
                f"[{indice_video:03d}/{total_videos:03d}] "
                f"{caminho_relativo} | "
                f"Real: {classe_real} | "
                f"Prevista: {classe_prevista}"
            )

        fps_global = (
            total_frames_modelo
            / tempo_total_modelo
            if tempo_total_modelo > 0
            else 0.0
        )

        metricas = calcular_metricas(
            classes_reais,
            classes_previstas,
        )

        salvar_resultados_videos(
            resultados_videos,
            pasta_modelo
            / "resultados_videos.csv",
        )

        gerar_matriz_confusao(
            classes_reais,
            classes_previstas,
            pasta_modelo
            / "matriz_confusao.png",
            pasta_modelo
            / "matriz_contagens.csv",
        )

        salvar_metricas(
            metricas,
            fps_global,
            pasta_modelo,
        )

        resumo_modelos.append(
            {
                "modelo": (
                    modelo_nome
                ),
                "acuracia": (
                    metricas[
                        "acuracia"
                    ]
                ),
                "precisao": (
                    metricas[
                        "precisao"
                    ]
                ),
                "recall": (
                    metricas[
                        "recall"
                    ]
                ),
                "f1_score": (
                    metricas[
                        "f1_score"
                    ]
                ),
                "fps": (
                    fps_global
                ),
            }
        )

        print()
        print(
            f"Acurácia: "
            f"{metricas['acuracia']:.3f}"
        )

        print(
            f"Precisão: "
            f"{metricas['precisao']:.3f}"
        )

        print(
            f"Recall: "
            f"{metricas['recall']:.3f}"
        )

        print(
            f"F1-score: "
            f"{metricas['f1_score']:.3f}"
        )

        print(
            f"FPS: "
            f"{fps_global:.3f}"
        )

        liberar_modelo(
            modelo
        )

    arquivo_resumo = (
        pasta_hardware
        / f"resultados_{hardware}.csv"
    )

    salvar_resumo_geral(
        resumo_modelos,
        arquivo_resumo,
    )

    arquivo_tabela = (
        pasta_hardware
        / f"tabela_metricas_{hardware.lower()}.png"
    )

    gerar_tabela_metricas_png(
        resumo_modelos,
        arquivo_tabela,
    )

    arquivo_grafico = (
        pasta_hardware
        / (
            "grafico_acuracia_fps_"
            f"{hardware.lower()}.png"
        )
    )

    gerar_grafico_acuracia_fps(
        resumo_modelos,
        hardware,
        arquivo_grafico,
    )

    print()
    print(
        "AVALIAÇÃO CONCLUÍDA"
    )

    print(
        f"Resultados salvos em: "
        f"{pasta_hardware}"
    )


if __name__ == "__main__":
    main()
