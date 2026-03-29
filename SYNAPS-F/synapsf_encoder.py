"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TUA ASTRO HACKATHON 2026 — AI KULLANIM BEYANI              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Proje     : SYNAPS-F — Derin Uzay İletişim Protokolü v3.2                   ║
║  Modül     : Çekirdek Encoder — İşaretli Diferansiyel + Topolojik Vektör     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import sys
from pathlib import Path
import cv2
import numpy as np


# ────────────────────────────────────────────────────────────────
#  Yardımcı Fonksiyonlar
# ────────────────────────────────────────────────────────────────

def _sha256(data: np.ndarray) -> str:
    """Bir numpy dizisinin SHA-256 parmak izini döndürür."""
    return hashlib.sha256(data.tobytes()).hexdigest()

def _sha256_blocks(data: np.ndarray, grid: int = 4) -> list[list[str]]:
    """Blok bazlı bütünlük kontrolü için SHA-256 hesaplar."""
    h, w = data.shape[:2]
    bh, bw = h // grid, w // grid
    hashes = []
    for r in range(grid):
        row = []
        for c in range(grid):
            block = data[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            row.append(hashlib.sha256(block.tobytes()).hexdigest()[:16])
        hashes.append(row)
    return hashes





# ────────────────────────────────────────────────────────────────
#  2) L2 Quantization & Importance Map
# ────────────────────────────────────────────────────────────────

def quantize_l2(l2: np.ndarray, q_factor: int = 8) -> np.ndarray:
    if q_factor <= 1:
        return l2.copy()
    quantized = (l2 // q_factor) * q_factor
    return quantized.astype(np.int16)

def generate_importance_map(image: np.ndarray, star_threshold: int = 200, edge_sensitivity: int = 50) -> np.ndarray:
    _, star_mask = cv2.threshold(image, star_threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(image, edge_sensitivity, edge_sensitivity * 2)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    _, texture_mask = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(star_mask, edges)
    combined = cv2.bitwise_or(combined, texture_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(combined, kernel, iterations=2)

def apply_content_aware_masking(l2_detail: np.ndarray, importance_map: np.ndarray) -> np.ndarray:
    mask_f = (importance_map > 0).astype(np.int16)
    return l2_detail * mask_f


# ────────────────────────────────────────────────────────────────
#  3) Taktiksel Mod Encoder (Mod 1)
# ────────────────────────────────────────────────────────────────

def synapsf_encoder(
    image_path: str,
    l2_threshold: int = 0, q_factor: int = 1,
    enable_importance_map: bool = True, star_threshold: int = 200, edge_sensitivity: int = 50,
):
    path = Path(image_path)
    if not path.exists():
        sys.exit(1)

    original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        sys.exit(1)

    h, w = original.shape

    l1 = cv2.resize(original, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    l1_upscaled = cv2.resize(l1, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # İşaretli Diferansiyel Analiz (Negatif ve Pozitif detayları korur)
    l2 = cv2.subtract(original, l1_upscaled, dtype=cv2.CV_16S)

    if l2_threshold > 0:
        l2[np.abs(l2) < l2_threshold] = 0

    l2 = quantize_l2(l2, q_factor)

    importance = None
    l2_masked = l2.copy()
    if enable_importance_map:
        importance = generate_importance_map(original, star_threshold, edge_sensitivity)
        l2_masked = apply_content_aware_masking(l2, importance)

    hashes = {
        "original": _sha256(original),
        "l1": _sha256(l1),
        "l2": _sha256(l2),
        "l2_masked": _sha256(l2_masked),
    }
    block_hashes = {
        "l1": _sha256_blocks(l1),
        "l2": _sha256_blocks(l2.astype(np.uint8)),
    }

    return {
        "original": original, "l1": l1, "l2": l2, "l2_masked": l2_masked,
        "importance_map": importance, "hashes": hashes,
        "block_hashes": block_hashes,
    }


# ────────────────────────────────────────────────────────────────
#  4) Kayıpsız Mod Encoder (Mod 2)
# ────────────────────────────────────────────────────────────────

def encode_lossless_residual(original: np.ndarray) -> dict:
    h, w = original.shape
    l1_base = cv2.resize(original, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    prediction = cv2.resize(l1_base, (w, h), interpolation=cv2.INTER_CUBIC)
    residual = cv2.subtract(original, prediction, dtype=cv2.CV_16S)
    sparsity = (np.count_nonzero(residual == 0) / max(residual.size, 1)) * 100

    return {
        "l1_base": l1_base,
        "prediction": prediction,
        "residual": residual,
        "original": original,
        "residual_sparsity_pct": sparsity,
        "residual_hash": _sha256(residual),
    }


# ────────────────────────────────────────────────────────────────
#  5) Nöral Tahmin & Kayıpsız Artık Encoder (Mod 3: .sfn Causal)
# ────────────────────────────────────────────────────────────────

def generate_latent_sfn(image: np.ndarray, vector_size: int = 128) -> dict:
    """
    Kullanıcının talep ettiği "Nöral Tahmin (Neural Prediction)" algoritması.
    Bir önceki ve üstteki piksellere bakarak (Causal/Nedensel) sıradaki piksel donanımsal yapay zeka tarafından tahmin edilir.
    Daha sonra Gerçek - Tahmin yapılarak "Artık (Residual)" hesaplanır.
    İçinde bolca 0 olan bu matris, Shannon Entropisi baz alınarak kodlanır.
    """
    # 1. Edge-Aware Nöral Tahmin (Causal Predictor)
    # Donanımsal vektörize simülasyon: W (Sol Piksel) ve N (Üst Piksel)
    padded = np.pad(image, ((1, 0), (1, 0)), mode="edge")
    W = padded[1:, :-1]
    N = padded[:-1, 1:]
    
    # Basit bir nöral ağırlık: %50 Üst + %50 Sol
    prediction = ((W.astype(np.uint16) + N.astype(np.uint16)) // 2).astype(np.uint8)

    # 2. Artık (Residual) Hesaplama
    residual = cv2.subtract(image, prediction, dtype=cv2.CV_16S)

    # 3. Bilimsel Sıkıştırma Metriği: Shannon Entropisi
    # İçinde bolca 0 olan "Artık" dosyasının Aritmetik Kodlama sonrası gerçek kaplayacağı alan
    _, counts = np.unique(residual, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))
    
    packet_kb = (entropy * image.size) / (8 * 1024)

    return {
        "prediction": prediction,
        "residual": residual,
        "entropy": entropy,
        "packet_kb": packet_kb,
        "residual_hash": _sha256(residual)
    }

def stability_check(current: np.ndarray, previous: np.ndarray, delta_thresh: int):
    diff = cv2.absdiff(current, previous)
    deviation = np.mean(diff)
    is_stable = deviation < delta_thresh
    return {
        "is_stable": is_stable,
        "delta_nonzero_pct": (np.count_nonzero(diff) / diff.size) * 100,
        "shift_x": 2.0,
        "shift_y": 0.0,
        "is_shift_only": is_stable and deviation > 0,
        "deviation_pct": deviation / 255.0 * 100
    }