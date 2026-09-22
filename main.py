import cv2
import mediapipe as mp
import math
import time
import threading
import numpy as np
from ultralytics import YOLO
import winsound

# Configurações gerais
PERIODO = 1 / 30

# Captura em alta para melhorar detecção ocular à distância
RESOLUCAO_CAPTURA = (1280, 720) #(1920, 1080) #

# Exibição leve
RESOLUCAO_EXIBICAO = (640, 480)

# Sono
TEMPO_CALIBRACAO = 2.5
TEMPO_FECHADO = 2.0
FRAMES_CONSECUTIVOS_FECHADO = 2

# Celular
PULAR_FRAMES_YOLO = 1.5
CONFIANCA_CELULAR = 0.50
YOLO_IMGSZ = 320

# Alarmes
COOLDOWN_ALARME_CELULAR = 0.05
COOLDOWN_ALARME_SONO = 0.05

# ROI facial focada na parte superior do rosto
FATOR_EXPANSAO_LATERAL = 0.30
FATOR_EXPANSAO_SUPERIOR = 0.35
FATOR_EXPANSAO_INFERIOR = 0.15
FATOR_ZOOM_FACE = 3.0

MIN_LARGURA_FACE = 80
MIN_ALTURA_FACE = 80

# Pré-processamento de baixa luz
ATIVAR_CLAHE = True
ATIVAR_GAMMA_ADAPTIVO = True
GAMMA_MIN = 1.2
GAMMA_MAX = 1.8
BRILHO_REFERENCIA = 110

# Suavização do EAR
ALPHA_EAR = 0.35

# Painéis
PAINEL_X = 12
PAINEL_SONO_Y1 = 12
PAINEL_SONO_Y2 = 55
PAINEL_CELULAR_Y1 = 65
PAINEL_CELULAR_Y2 = 108

# Índices para EAR do olho direito
P_DIR_H1 = 33
P_DIR_H2 = 133
P_DIR_V1A = 160
P_DIR_V1B = 144
P_DIR_V2A = 159
P_DIR_V2B = 145

# Índices para EAR do olho esquerdo
P_ESQ_H1 = 362
P_ESQ_H2 = 263
P_ESQ_V1A = 385
P_ESQ_V1B = 380
P_ESQ_V2A = 386
P_ESQ_V2B = 374

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Estado global
controle_alarme_celular = False
controle_alarme_sono = False

status_olhos = "A"
inicio_fechado = None
contador_fechado = 0

ultima_caixa_celular = []
contador_frames = 0

ultimo_alarme_celular = 0.0
ultimo_alarme_sono = 0.0

calibrando = True
inicio_calibracao = time.time()

ear_amostras_dir = []
ear_amostras_esq = []

ear_base_dir = None
ear_base_esq = None
ear_limiar_dir = None
ear_limiar_esq = None

ear_filtrado_dir = None
ear_filtrado_esq = None

# Inicialização da câmera
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
video.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUCAO_CAPTURA[0])
video.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCAO_CAPTURA[1])
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# YOLO
modelo = YOLO("yolov8m.pt")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

faceMesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

faceDetection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.45
)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def alarme_celular():
    global controle_alarme_celular

    if controle_alarme_celular or controle_alarme_sono:
        return

    controle_alarme_celular = True
    try:
        winsound.Beep(1800, 250)
    finally:
        controle_alarme_celular = False


def alarme_sono():
    global controle_alarme_sono

    if controle_alarme_sono:
        return

    controle_alarme_sono = True
    try:
        winsound.Beep(2500, 180)
        winsound.Beep(2500, 180)
    finally:
        controle_alarme_sono = False


def iniciar_alarme_celular():
    global ultimo_alarme_celular

    agora = time.time()
    if controle_alarme_sono:
        return

    if agora - ultimo_alarme_celular >= COOLDOWN_ALARME_CELULAR and not controle_alarme_celular:
        ultimo_alarme_celular = agora
        threading.Thread(target=alarme_celular, daemon=True).start()


def iniciar_alarme_sono():
    global ultimo_alarme_sono

    agora = time.time()
    if agora - ultimo_alarme_sono >= COOLDOWN_ALARME_SONO and not controle_alarme_sono:
        ultimo_alarme_sono = agora
        threading.Thread(target=alarme_sono, daemon=True).start()


def detectar_celular(img_bgr):
    resultado = modelo.predict(
        source=img_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONFIANCA_CELULAR,
        classes=[67],
        verbose=False,
        device="cpu"
    )

    melhor_caixa = []
    melhor_conf = 0.0

    for objetos in resultado:
        if objetos.boxes is None or len(objetos.boxes) == 0:
            continue

        for dados in objetos.boxes:
            conf = float(dados.conf[0])
            if conf > melhor_conf:
                x1, y1, x2, y2 = map(int, dados.xyxy[0])
                melhor_caixa = [x1, y1, x2, y2]
                melhor_conf = conf

    return melhor_caixa


def aplicar_gamma(img_bgr, gamma):
    if gamma <= 0:
        return img_bgr

    inv_gamma = 1.0 / gamma
    tabela = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(img_bgr, tabela)


def aplicar_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocessar_roi_face(img_bgr):
    saida = img_bgr.copy()

    if ATIVAR_GAMMA_ADAPTIVO:
        gray = cv2.cvtColor(saida, cv2.COLOR_BGR2GRAY)
        media = gray.mean()

        if media < BRILHO_REFERENCIA:
            intensidade = (BRILHO_REFERENCIA - media) / BRILHO_REFERENCIA
            gamma = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * intensidade

            if gamma < GAMMA_MIN:
                gamma = GAMMA_MIN
            if gamma > GAMMA_MAX:
                gamma = GAMMA_MAX

            saida = aplicar_gamma(saida, gamma)

    if ATIVAR_CLAHE:
        saida = aplicar_clahe_bgr(saida)

    return saida


def detectar_face_roi_superior(img_bgr):
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    det = faceDetection.process(img_rgb)

    if not det or not det.detections:
        return None

    melhor = det.detections[0]
    caixa = melhor.location_data.relative_bounding_box

    x = int(caixa.xmin * w)
    y = int(caixa.ymin * h)
    bw = int(caixa.width * w)
    bh = int(caixa.height * h)

    if bw < MIN_LARGURA_FACE or bh < MIN_ALTURA_FACE:
        return None

    x1 = int(max(0, x - bw * FATOR_EXPANSAO_LATERAL))
    x2 = int(min(w, x + bw + bw * FATOR_EXPANSAO_LATERAL))

    y1 = int(max(0, y - bh * FATOR_EXPANSAO_SUPERIOR))
    y2 = int(min(h, y + bh * (0.50 + FATOR_EXPANSAO_INFERIOR)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def ponto_px(face, idx, largura, altura):
    x = int(face.landmark[idx].x * largura)
    y = int(face.landmark[idx].y * altura)
    return (x, y)


def distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calcular_ear_olho(face, largura, altura, h1, h2, v1a, v1b, v2a, v2b):
    ph1 = ponto_px(face, h1, largura, altura)
    ph2 = ponto_px(face, h2, largura, altura)
    pv1a = ponto_px(face, v1a, largura, altura)
    pv1b = ponto_px(face, v1b, largura, altura)
    pv2a = ponto_px(face, v2a, largura, altura)
    pv2b = ponto_px(face, v2b, largura, altura)

    horizontal = distancia(ph1, ph2)
    vertical1 = distancia(pv1a, pv1b)
    vertical2 = distancia(pv2a, pv2b)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def suavizar_ear(valor_atual, valor_anterior):
    if valor_anterior is None:
        return valor_atual
    return ALPHA_EAR * valor_atual + (1.0 - ALPHA_EAR) * valor_anterior


def mapear_ponto_roi_para_tela(x_roi, y_roi, x1_roi, y1_roi, largura_zoom, altura_zoom, largura_face, altura_face, sx, sy):
    x_face = int((x_roi / largura_zoom) * largura_face)
    y_face = int((y_roi / altura_zoom) * altura_face)

    x_cap = x1_roi + x_face
    y_cap = y1_roi + y_face

    return int(x_cap * sx), int(y_cap * sy)


def desenhar_pontos_olhos(img_exibicao, face, largura_zoom, altura_zoom, x1_roi, y1_roi, largura_face, altura_face, sx, sy):
    for idx in LEFT_EYE + RIGHT_EYE:
        px, py = ponto_px(face, idx, largura_zoom, altura_zoom)
        ex, ey = mapear_ponto_roi_para_tela(
            px, py, x1_roi, y1_roi, largura_zoom, altura_zoom,
            largura_face, altura_face, sx, sy
        )
        cv2.circle(img_exibicao, (ex, ey), 1, (0, 255, 255), -1)

    pares = [
        (P_DIR_V1A, P_DIR_V1B, (255, 0, 0), 2),
        (P_DIR_V2A, P_DIR_V2B, (255, 0, 0), 2),
        (P_DIR_H1, P_DIR_H2, (0, 255, 0), 1),
        (P_ESQ_V1A, P_ESQ_V1B, (255, 0, 0), 2),
        (P_ESQ_V2A, P_ESQ_V2B, (255, 0, 0), 2),
        (P_ESQ_H1, P_ESQ_H2, (0, 255, 0), 1)
    ]

    for a, b, cor, esp in pares:
        p1x, p1y = ponto_px(face, a, largura_zoom, altura_zoom)
        p2x, p2y = ponto_px(face, b, largura_zoom, altura_zoom)

        e1x, e1y = mapear_ponto_roi_para_tela(
            p1x, p1y, x1_roi, y1_roi, largura_zoom, altura_zoom,
            largura_face, altura_face, sx, sy
        )
        e2x, e2y = mapear_ponto_roi_para_tela(
            p2x, p2y, x1_roi, y1_roi, largura_zoom, altura_zoom,
            largura_face, altura_face, sx, sy
        )

        cv2.line(img_exibicao, (e1x, e1y), (e2x, e2y), cor, esp)


def desenhar_area_roi(img_exibicao, x1_roi, y1_roi, x2_roi, y2_roi, sx, sy):
    ex1 = int(x1_roi * sx)
    ey1 = int(y1_roi * sy)
    ex2 = int(x2_roi * sx)
    ey2 = int(y2_roi * sy)
    cv2.rectangle(img_exibicao, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)


def processar_sonolencia(frame_captura_bgr, img_exibicao):
    global status_olhos
    global inicio_fechado
    global contador_fechado
    global calibrando
    global ear_amostras_dir
    global ear_amostras_esq
    global ear_base_dir
    global ear_base_esq
    global ear_limiar_dir
    global ear_limiar_esq
    global ear_filtrado_dir
    global ear_filtrado_esq

    sx = RESOLUCAO_EXIBICAO[0] / RESOLUCAO_CAPTURA[0]
    sy = RESOLUCAO_EXIBICAO[1] / RESOLUCAO_CAPTURA[1]

    roi = detectar_face_roi_superior(frame_captura_bgr)
    if roi is None:
        status_olhos = "A"
        inicio_fechado = None
        contador_fechado = 0
        return

    x1, y1, x2, y2 = roi
    face_roi = frame_captura_bgr[y1:y2, x1:x2]

    if face_roi.size == 0:
        status_olhos = "A"
        inicio_fechado = None
        contador_fechado = 0
        return

    largura_face = x2 - x1
    altura_face = y2 - y1

    face_roi = preprocessar_roi_face(face_roi)

    face_zoom = cv2.resize(
        face_roi,
        (int(largura_face * FATOR_ZOOM_FACE), int(altura_face * FATOR_ZOOM_FACE)),
        interpolation=cv2.INTER_CUBIC
    )

    face_zoom_rgb = cv2.cvtColor(face_zoom, cv2.COLOR_BGR2RGB)
    results_face = faceMesh.process(face_zoom_rgb)

    if not results_face or not results_face.multi_face_landmarks:
        status_olhos = "A"
        inicio_fechado = None
        contador_fechado = 0
        return

    face = results_face.multi_face_landmarks[0]
    hz, wz, _ = face_zoom.shape

    desenhar_area_roi(img_exibicao, x1, y1, x2, y2, sx, sy)
    desenhar_pontos_olhos(img_exibicao, face, wz, hz, x1, y1, largura_face, altura_face, sx, sy)

    ear_dir = calcular_ear_olho(face, wz, hz, P_DIR_H1, P_DIR_H2, P_DIR_V1A, P_DIR_V1B, P_DIR_V2A, P_DIR_V2B)
    ear_esq = calcular_ear_olho(face, wz, hz, P_ESQ_H1, P_ESQ_H2, P_ESQ_V1A, P_ESQ_V1B, P_ESQ_V2A, P_ESQ_V2B)

    ear_filtrado_dir = suavizar_ear(ear_dir, ear_filtrado_dir)
    ear_filtrado_esq = suavizar_ear(ear_esq, ear_filtrado_esq)

    if calibrando:
        ear_amostras_dir.append(ear_filtrado_dir)
        ear_amostras_esq.append(ear_filtrado_esq)

        if time.time() - inicio_calibracao >= TEMPO_CALIBRACAO:
            if ear_amostras_dir and ear_amostras_esq:
                ear_base_dir = sum(ear_amostras_dir) / len(ear_amostras_dir)
                ear_base_esq = sum(ear_amostras_esq) / len(ear_amostras_esq)

                ear_limiar_dir = ear_base_dir * 0.72
                ear_limiar_esq = ear_base_esq * 0.72

                if ear_limiar_dir < 0.16:
                    ear_limiar_dir = 0.16
                if ear_limiar_dir > 0.30:
                    ear_limiar_dir = 0.30

                if ear_limiar_esq < 0.16:
                    ear_limiar_esq = 0.16
                if ear_limiar_esq > 0.30:
                    ear_limiar_esq = 0.30
            else:
                ear_base_dir = 0.24
                ear_base_esq = 0.24
                ear_limiar_dir = 0.18
                ear_limiar_esq = 0.18

            calibrando = False

        status_olhos = "A"
        inicio_fechado = None
        contador_fechado = 0
        return

    olho_dir_fechado = ear_filtrado_dir <= ear_limiar_dir
    olho_esq_fechado = ear_filtrado_esq <= ear_limiar_esq

    olhos_fechados = olho_dir_fechado and olho_esq_fechado

    if olhos_fechados:
        contador_fechado += 1
    else:
        contador_fechado = 0
        status_olhos = "A"
        inicio_fechado = None

    if contador_fechado >= FRAMES_CONSECUTIVOS_FECHADO:
        if inicio_fechado is None:
            inicio_fechado = time.time()

        status_olhos = "F"

        if time.time() - inicio_fechado >= TEMPO_FECHADO:
            iniciar_alarme_sono()
        else:
            status_olhos = "A"


def desenhar_status_sono(img):
    if calibrando:
        cv2.rectangle(img, (PAINEL_X, PAINEL_SONO_Y1), (270, PAINEL_SONO_Y2), (0, 140, 255), -1)
        cv2.putText(img, 'CALIBRANDO OLHOS', (PAINEL_X + 10, PAINEL_SONO_Y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    elif status_olhos == "F":
        cv2.rectangle(img, (PAINEL_X, PAINEL_SONO_Y1), (245, PAINEL_SONO_Y2), (0, 0, 255), -1)
        cv2.putText(img, 'ALERTA SONO', (PAINEL_X + 10, PAINEL_SONO_Y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(img, (PAINEL_X, PAINEL_SONO_Y1), (195, PAINEL_SONO_Y2), (0, 170, 0), -1)
        cv2.putText(img, 'ACORDADO', (PAINEL_X + 10, PAINEL_SONO_Y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def desenhar_status_celular(img, celular):
    if celular:
        x1, y1, x2, y2 = celular

        cv2.rectangle(img, (PAINEL_X, PAINEL_CELULAR_Y1), (255, PAINEL_CELULAR_Y2), (0, 0, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, 'ALERTA CELULAR', (PAINEL_X + 10, PAINEL_CELULAR_Y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

        if status_olhos != "F" and not calibrando:
            iniciar_alarme_celular()
    else:
        cv2.rectangle(img, (PAINEL_X, PAINEL_CELULAR_Y1), (220, PAINEL_CELULAR_Y2), (0, 170, 0), -1)
        cv2.putText(img, 'SEM CELULAR', (PAINEL_X + 10, PAINEL_CELULAR_Y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


while True:
    ciclo_inicio = time.time()

    ret, frame_captura = video.read()
    if not ret:
        break

    frame_captura = cv2.resize(frame_captura, RESOLUCAO_CAPTURA)
    img_exibicao = cv2.resize(frame_captura, RESOLUCAO_EXIBICAO)

    processar_sonolencia(frame_captura, img_exibicao)

    if contador_frames % PULAR_FRAMES_YOLO == 0:
        ultima_caixa_celular = detectar_celular(img_exibicao)

    desenhar_status_sono(img_exibicao)
    desenhar_status_celular(img_exibicao, ultima_caixa_celular)

    cv2.imshow('img', img_exibicao)

    contador_frames += 1

    if cv2.waitKey(1) == 27:
        break

    tempo_execucao = time.time() - ciclo_inicio
    if tempo_execucao < PERIODO:
        time.sleep(PERIODO - tempo_execucao)

video.release()
cv2.destroyAllWindows()