"""Fuente de imágenes para Raspberry Pi Zero 2 W + cámara de 8 Mpx (IMX219).

Implementa tres técnicas de la tabla:
  T1  Recorte del sensor (ROI de captura) con ``ScalerCrop``.
  T2  Exposición, ganancia y balance de blancos fijos (AE y AWB apagados).
  T3  Uso directo del canal Y del stream YUV420 (sin conversión de color).

Picamera2 se instala con apt (``sudo apt install python3-picamera2``); por eso
se importa de forma diferida y el resto del proyecto funciona sin ella en un PC.
"""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from ..config import CameraCfg
from .strip import orient_strip


class PiCameraSource:
    """Entrega ``(frame_idx, timestamp_s, franja_Y)`` a la tasa del sensor."""

    def __init__(self, cam: CameraCfg):
        from picamera2 import Picamera2  # import diferido: sólo existe en la Pi

        self.cam = cam
        self.picam2 = Picamera2()

        # --- T1: elegir modo de sensor rápido y calcular el recorte -----------
        out_w = cam.output_width
        out_h = cam.strip_rows
        if cam.flow_axis == "x":  # la franja es una banda vertical del sensor
            out_w, out_h = cam.strip_rows, cam.output_width

        config = self.picam2.create_video_configuration(
            main={"size": (out_w, out_h), "format": "YUV420"},
            sensor={"output_size": tuple(cam.sensor_mode_size)},
            buffer_count=cam.buffer_count,
            controls={
                "FrameDurationLimits": (int(1e6 / cam.fps), int(1e6 / cam.fps)),
                # --- T2: todo fijo -> el fondo retroiluminado siempre mide igual
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": int(cam.exposure_us),
                "AnalogueGain": float(cam.analogue_gain),
                "ColourGains": tuple(cam.colour_gains),
            },
        )
        self.picam2.align_configuration(config)  # el ISP exige anchos alineados
        self.picam2.configure(config)
        self.size = config["main"]["size"]  # (w, h) reales tras alinear

        # T2 (extensión): cámaras con autofoco (IMX708 / Camera Module 3).
        # El enfoque también debe quedar FIJO: si el autofoco "busca" cuando pasa
        # un pez, cambia la nitidez del borde y con ella el área segmentada.
        if "AfMode" in self.picam2.camera_controls:
            from libcamera import controls as lc
            self.picam2.set_controls({"AfMode": lc.AfModeEnum.Manual,
                                      "LensPosition": float(cam.lens_position)})

        # ScalerCrop se expresa en coordenadas del sensor completo. El máximo
        # permitido depende del modo elegido (p. ej. 3072x1728 centrado para el
        # modo 1536x864 del IMX708).
        x0, y0, w0, h0 = self.picam2.camera_properties.get(
            "ScalerCropMaximum", self.picam2.camera_controls["ScalerCrop"][1])
        if cam.flow_axis == "y":
            # Franja horizontal: todo el ancho, pocas filas. Se mantiene el
            # píxel cuadrado: filas_sensor = filas_salida * (w0 / ancho_salida)
            hs = int(round(self.size[1] * w0 / self.size[0]))
            ys = int(y0 + cam.strip_center_frac * h0 - hs / 2)
            ys = max(y0, min(y0 + h0 - hs, ys))
            crop = (x0, ys, w0, hs)
        else:
            ws = int(round(self.size[0] * h0 / self.size[1]))
            xs = int(x0 + cam.strip_center_frac * w0 - ws / 2)
            xs = max(x0, min(x0 + w0 - ws, xs))
            crop = (xs, y0, ws, h0)
        self.crop = crop
        self.picam2.set_controls({"ScalerCrop": crop})
        self.frame_period_ns = int(1e9 / cam.fps)

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        self.picam2.start()
        time.sleep(0.5)  # deja que los controles fijos se apliquen
        t0_ns = None
        try:
            while True:
                req = self.picam2.capture_request()
                try:
                    yuv = req.make_array("main")
                    md = req.get_metadata()
                finally:
                    req.release()
                w, h = self.size
                # --- T3: en YUV420 las primeras h filas son el plano Y ---------
                y = yuv[:h, :w]
                ts = md.get("SensorTimestamp", time.monotonic_ns())
                if t0_ns is None:
                    t0_ns = ts
                # El índice se deriva del timestamp: si el sistema pierde un
                # cuadro el pipeline lo sabe y no deforma la imagen espacio-temporal.
                idx = int(round((ts - t0_ns) / self.frame_period_ns))
                yield idx, (ts - t0_ns) / 1e9, orient_strip(y, self.cam)
        finally:
            self.picam2.stop()

    def close(self) -> None:
        self.picam2.close()
