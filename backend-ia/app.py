from flask import Flask, request, jsonify

import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

app = Flask(__name__)

# ================= MODELO =================

MODEL_PATH = "models/face_landmarker.task"

base_options = BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)

# ================= ROTA TESTE =================

@app.route('/')
def home():

    return jsonify({
        'status': 'API DNP ONLINE'
    })

# ================= DETECÇÃO =================

@app.route('/detectar', methods=['POST'])
def detectar():

    try:

        if 'foto' not in request.files:
            return jsonify({
                'erro': 'Foto não enviada'
            }), 400

        arquivo = request.files['foto']

        imagem_bytes = np.frombuffer(
            arquivo.read(),
            np.uint8
        )

        imagem = cv2.imdecode(
            imagem_bytes,
            cv2.IMREAD_COLOR
        )

        if imagem is None:
            return jsonify({
                'erro': 'Imagem inválida'
            }), 400

        rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )

        altura, largura, _ = imagem.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        resultado = detector.detect(mp_image)

        if not resultado.face_landmarks:
            return jsonify({
                'erro': 'Rosto não detectado'
            }), 400

        face = resultado.face_landmarks[0]

        # Pupilas reais
        pupila_esquerda = face[468]
        pupila_direita = face[473]

        x1 = int(pupila_esquerda.x * largura)
        y1 = int(pupila_esquerda.y * altura)

        x2 = int(pupila_direita.x * largura)
        y2 = int(pupila_direita.y * altura)

        # Distância entre pupilas
        distancia_pixels = abs(x2 - x1)

        # Escala temporária
        mm_por_pixel = 0.13

        # DNP total
        dnp = round(
            distancia_pixels * mm_por_pixel,
            1
        )

        # Centro facial
        centro = largura / 2

        # Monocular
        od = round(
            abs(centro - x1) * mm_por_pixel,
            1
        )

        oe = round(
            abs(x2 - centro) * mm_por_pixel,
            1
        )

        return jsonify({
            'dnp': dnp,
            'od': od,
            'oe': oe,
            'pupilas': {
                'esquerda': {
                    'x': x1,
                    'y': y1
                },
                'direita': {
                    'x': x2,
                    'y': y2
                }
            },
            'status': 'Pupilas reais detectadas'
        })

    except Exception as e:

        return jsonify({
            'erro': str(e)
        }), 500

# ================= START =================

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )