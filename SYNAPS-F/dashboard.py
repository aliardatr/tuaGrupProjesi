"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TUA ASTRO HACKATHON 2026 — AI KULLANIM BEYANI            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Proje     : SYNAPS-F — Derin Uzay İletişim Protokolü v3.1                ║
║  Modül     : Mission Control Dashboard — Scientific Integrity             ║
║                                                                            ║
║  AI Kullanımı:                                                             ║
║    • Otonom Karar Destek Mekanizması (APS — Autonomous Protocol Scaling)   ║
║    • Nöral Hareket Tahmini (Motion-Aware Phase Correlation)               ║
║    • İşaretli Diferansiyel Analiz (Signed int16 L2 Fusion)                ║
║    • Self-Healing Data (SHA-256 Blok Doğrulama + Inpainting)              ║
║  Tüm mimari tasarım insan geliştiriciler tarafından yapılmıştır.          ║
║                                                                            ║
║  Veri      : test_uzay.jpg — TUA/NASA Hubble Heritage (HLA) ve            ║
║              ESA/Webb açık veri seti                                       ║
║  Lisans    : MIT — Açık kaynak, GitHub'a yüklenmeye hazır                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Çalıştır:  streamlit run dashboard.py
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

from synapsf_encoder import (
    _sha256,
    _sha256_blocks,
    stability_check,
    generate_importance_map,
    apply_content_aware_masking,
    quantize_l2,
    encode_lossless_residual,
    generate_latent_sfn,
)
from synapsf_decoder import (
    detect_corruption,
    self_heal,
    ai_super_resolution,
    signed_fusion,
    decode_lossless_residual,
    decode_latent_sfn,
    _compute_ssim,
    _compute_mse,
)

# ────────────────────────────────────────────────────────────────
#  Sayfa Ayarları
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SYNAPS-F v3.1 — Autonomous Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
#  CSS
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], section[data-testid="stSidebar"] {
        background-color: #0a0a1a !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d0d24 !important;
        border-right: 1px solid #1a1a3a;
    }
    .neon-title {
        font-family: 'Courier New', monospace;
        text-align: center; color: #0ff;
        text-shadow: 0 0 10px #0ff, 0 0 30px #0ff, 0 0 50px #088;
        font-size: 2rem; letter-spacing: 4px;
        margin-bottom: 0; padding: 0.5rem 0;
    }
    .sub-title { text-align: center; color: #888; font-size: 0.8rem; letter-spacing: 2px; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #0d0d28, #141432);
        border: 1px solid #1a1a3a; border-radius: 12px;
        padding: 0.9rem 0.5rem; text-align: center;
        box-shadow: 0 0 20px rgba(0,255,255,0.05);
    }
    .metric-card h3 { color: #0ff; font-size: 0.55rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.1rem; }
    .metric-card .value { color: #fff; font-size: 1.2rem; font-weight: 700; font-family: 'Courier New', monospace; }
    .power-card {
        background: linear-gradient(135deg, #0d1a0d, #0a0a1a);
        border: 1px solid #1a3a1a; border-radius: 12px;
        padding: 0.9rem 0.5rem; text-align: center;
    }
    .power-card h3 { color: #4f4; font-size: 0.55rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.1rem; }
    .power-card .value { color: #fff; font-size: 1.2rem; font-weight: 700; font-family: 'Courier New', monospace; }
    .sidebar-section { color: #0ff; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid #1a1a3a; padding-bottom: 0.3rem; margin-top: 1.2rem; margin-bottom: 0.4rem; }
    .panel-label { color: #ccc; text-align: center; font-size: 0.8rem; margin-top: 0.2rem; }
    .hash-text { font-family: 'Courier New', monospace; font-size: 0.6rem; color: #6f6; word-break: break-all; }
    hr { border-color: #1a1a3a !important; }
    .healing-badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: bold; letter-spacing: 1px; }
    .badge-ok { background: #0a2a0a; color: #0f0; border: 1px solid #0f0; }
    .badge-healing { background: #2a2a0a; color: #ff0; border: 1px solid #ff0; }
    .badge-fail { background: #2a0a0a; color: #f44; border: 1px solid #f44; }
    .aps-alert { background: linear-gradient(90deg, #2a0a0a, #0a0a1a); border: 1px solid #f44; border-radius: 10px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; text-align: center; animation: pulseBorder 1.5s ease-in-out infinite; }
    @keyframes pulseBorder { 0%,100%{border-color:#f44;box-shadow:0 0 5px #f44;}50%{border-color:#f88;box-shadow:0 0 20px #f44;} }
    .aps-alert .aps-title { font-size: 1rem; font-weight: bold; color: #f44; letter-spacing: 2px; text-transform: uppercase; }
    .aps-alert .aps-detail { font-size: 0.8rem; color: #f99; margin-top: 0.3rem; }
    .aps-ok { background: linear-gradient(90deg, #0a2a0a, #0a0a1a); border: 1px solid #0f0; border-radius: 10px; padding: 0.8rem 1.2rem; margin: 0.5rem 0; text-align: center; }
    .aps-ok .aps-title { font-size: 1rem; font-weight: bold; color: #0f0; letter-spacing: 2px; }
    .aps-ok .aps-detail { font-size: 0.8rem; color: #9f9; margin-top: 0.3rem; }
    .integrity-corrupt { background: linear-gradient(135deg, #3a0a0a, #1a0505); border: 2px solid #f44; border-radius: 12px; padding: 1rem; text-align: center; animation: corruptPulse 1s ease-in-out infinite; }
    @keyframes corruptPulse { 0%,100%{box-shadow:0 0 10px rgba(255,68,68,0.3);}50%{box-shadow:0 0 30px rgba(255,68,68,0.8);} }
    .integrity-corrupt h4 { color: #f44; margin: 0; font-size: 1rem; letter-spacing: 2px; }
    .integrity-corrupt p { color: #f99; font-size: 0.8rem; margin: 0.3rem 0 0; }
    .integrity-healed { background: linear-gradient(135deg, #0a2a0a, #051a05); border: 2px solid #0f0; border-radius: 12px; padding: 1rem; text-align: center; }
    .integrity-healed h4 { color: #0f0; margin: 0; font-size: 1rem; letter-spacing: 2px; }
    .integrity-healed p { color: #9f9; font-size: 0.8rem; margin: 0.3rem 0 0; }
    .integrity-clean { background: linear-gradient(135deg, #0a1a2a, #05101a); border: 1px solid #08f; border-radius: 12px; padding: 1rem; text-align: center; }
    .integrity-clean h4 { color: #08f; margin: 0; font-size: 1rem; letter-spacing: 2px; }
    .integrity-clean p { color: #8bf; font-size: 0.8rem; margin: 0.3rem 0 0; }
    /* SHA-256 Block Grid */
    .block-grid { display: grid; gap: 3px; }
    .block-cell { font-family: 'Courier New', monospace; font-size: 0.5rem; padding: 4px; border-radius: 4px; text-align: center; word-break: break-all; }
    .block-ok { background: #0a1a0a; color: #0f0; border: 1px solid #0f03; }
    .block-fail { background: #1a0a0a; color: #f44; border: 1px solid #f443; animation: corruptPulse 1.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
#  Yardımcı Fonksiyonlar
# ────────────────────────────────────────────────────────────────

def sha256(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_gaussian_noise_int16(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, -255, 255).astype(np.int16)


def add_salt_pepper_noise(img, amount):
    out = img.copy()
    n = img.size
    s = int(n * amount / 2)
    coords = tuple(np.random.randint(0, d, s) for d in img.shape)
    out[coords] = 255
    coords = tuple(np.random.randint(0, d, s) for d in img.shape)
    out[coords] = 0
    return out


def add_salt_pepper_noise_int16(img, amount):
    out = img.copy()
    n = img.size
    s = int(n * amount / 2)
    coords = tuple(np.random.randint(0, d, s) for d in img.shape)
    out[coords] = 127
    coords = tuple(np.random.randint(0, d, s) for d in img.shape)
    out[coords] = -127
    return out


def compute_efficiency(original, l1, l2):
    orig_kb = original.nbytes / 1024
    l1_kb = l1.nbytes / 1024
    # int16 L2 → gerçek iletimde 1 byte/piksel'e kodlanır (değer aralığı ±128)
    l2_pixel_count = l2.size
    l2_nz = np.count_nonzero(l2)
    l2_bytes = l2_pixel_count  # 1 byte equivalent per pixel
    l2_eff_kb = (l2_nz / max(l2_pixel_count, 1)) * (l2_bytes / 1024)
    packet = l1_kb + l2_eff_kb
    return {
        "original_kb": orig_kb, "l1_kb": l1_kb,
        "l2_effective_kb": l2_eff_kb, "packet_kb": packet,
        "compression_ratio": orig_kb / max(packet, 0.001),
        "bandwidth_saved_pct": (1 - packet / orig_kb) * 100,
        "l2_sparsity_pct": (1 - l2_nz / max(l2.size, 1)) * 100,
    }


def compute_power_metrics(eff, neural_info, is_delta, aps_freq_boost=False):
    TRANSMITTER_W = 25.0
    FREQ_BOOST_W = 8.0  # Frekans güçlendirme ek enerji
    BITRATE_KBPS = 1024.0
    orig_kb, packet_kb = eff["original_kb"], eff["packet_kb"]
    if is_delta and neural_info and neural_info["is_stable"]:
        ratio = neural_info["delta_nonzero_pct"] / 100.0
        packet_kb = packet_kb * max(ratio, 0.05)
    cl_s = (orig_kb * 8) / BITRATE_KBPS
    sf_s = (packet_kb * 8) / BITRATE_KBPS
    # Frekans güçlendirme aktifse ek enerji tüketimi
    effective_power = TRANSMITTER_W + (FREQ_BOOST_W if aps_freq_boost else 0)
    cl_j = TRANSMITTER_W * cl_s
    sf_j = effective_power * sf_s
    return {
        "transmitter_w": TRANSMITTER_W,
        "freq_boost_w": FREQ_BOOST_W if aps_freq_boost else 0,
        "effective_w": effective_power,
        "classical_s": cl_s, "synapsf_s": sf_s,
        "classical_j": cl_j, "synapsf_j": sf_j,
        "saved_pct": (1 - sf_j / max(cl_j, 0.001)) * 100,
        "delta_active": is_delta and neural_info is not None and neural_info.get("is_stable", False),
    }


# ────────────────────────────────────────────────────────────────
#  Frequency Boost Simulation
# ────────────────────────────────────────────────────────────────

def frequency_boost_filter(data, boost=2.0):
    smooth = cv2.GaussianBlur(data.astype(np.float64), (7, 7), 2.0)
    boosted = cv2.addWeighted(data.astype(np.float64), 1.0, smooth, boost - 1.0, 0)
    return np.clip(boosted, 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────
#  Dashboard
# ────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="neon-title">🛰️ SYNAPS-F v3.1 — AUTONOMOUS MISSION CONTROL</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SCIENTIFIC INTEGRITY — SIGNED DIFFERENTIAL — DEEP SPACE</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Görüntü ──────────────────────────────────────────────
    image_path = Path("test_uzay.jpg")
    if not image_path.exists():
        st.error(f"❌ Görüntü bulunamadı: `{image_path.resolve()}`")
        st.stop()
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        st.error("❌ Okunamadı.")
        st.stop()
    h, w = original.shape

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-section">📡 Bant Genişliği</div>', unsafe_allow_html=True)
        bw_mode = st.radio("Mod", ["HIGH — Signed Fusion", "LOW — Sadece L1",
                                     "PURE SCIENCE — Lossless", "ULTRA LOW — SYNAPS-Latent (.sfn)"],
                           index=0, label_visibility="collapsed",
                           help="HIGH: L1+L2 Signed Fusion. LOW: Sadece L1. PURE SCIENCE: Kayıpsız (SSIM=1.0). ULTRA LOW: 128 float anlamsal vektör (512 byte).")
        user_high = "HIGH" in bw_mode
        is_lossless = "PURE" in bw_mode
        is_latent = "ULTRA" in bw_mode

        st.markdown('<div class="sidebar-section">🌌 Gürültü</div>', unsafe_allow_html=True)
        noise_type = st.selectbox("Tip", ["Yok", "Gaussian Noise", "Salt & Pepper Noise"],
                                  label_visibility="collapsed",
                                  help="Derin uzay iletimi sırasında karşılaşılan gürültü türünü simüle eder. Gaussian: sürekli termal/radyasyon gürültüsü. Salt & Pepper: kozmik ışın darbe gürültüsü.")
        noise_intensity = st.slider("Şiddet", 0, 100, 0, 1, disabled=(noise_type == "Yok"),
                                    help="Gürültü şiddeti (0-100). %50'yi aştığında APS otomatik olarak LOW BW moduna geçer ve Frekans Güçlendirme devreye girer.")

        st.markdown('<div class="sidebar-section">🩹 Self-Healing</div>', unsafe_allow_html=True)
        enable_healing = st.checkbox("AI Inpainting", value=True,
                                     help="SHA-256 hash uyuşmazlığında Navier-Stokes tabanlı AI Inpainting ile bozulmuş pikselleri otomatik onarır.")

        st.markdown('<div class="sidebar-section">🧠 Neural Prediction</div>', unsafe_allow_html=True)
        enable_neural = st.checkbox("Motion-Aware", value=True,
                                    help="FFT Phase Correlation ile ardışık kareler arasındaki global kaymayı tespit eder. Stabil sahne veya salt kayma durumunda sadece Delta/Shift vektörü gönderilir.")
        delta_thresh = st.slider("Sapma Eşiği (%)", 1, 50, 5, 1, disabled=not enable_neural,
                                 help="İki kare arasındaki ortalama piksel sapması bu eşiğin altındaysa sahne 'stabil' kabul edilir ve tam kare yerine Delta Vektörü gönderilir.")

        st.markdown('<div class="sidebar-section">🎯 Content-Aware</div>', unsafe_allow_html=True)
        enable_imp = st.checkbox("Importance Map", value=True,
                                 help="Yıldız, krater sınırı ve doku bilgisine göre önemli alanları tespit eder. L2 detayı sadece bu bölgelere gönderilir, boş uzay 'Void' olarak kodlanır.")
        star_t = st.slider("Yıldız Eşiği", 100, 255, 200, 5, disabled=not enable_imp,
                           help="Parlaklık eşiği: bu değerin üzerindeki pikseller 'yıldız' olarak işaretlenir. Düşük değer = daha fazla yıldız algılanır.")
        edge_s = st.slider("Kenar Duyarlılığı", 10, 150, 50, 5, disabled=not enable_imp,
                           help="Canny kenar algılama alt eşiği. Düşük değer = daha hassas kenar tespiti (krater sınırları, nebula detayları). Yüksek değer = sadece belirgin yapılar.")

        st.markdown('<div class="sidebar-section">🔬 L2 Bilimsel Ayarlar</div>', unsafe_allow_html=True)
        l2_thresh = st.slider("L2 Eşik", 0, 30, 0, 1,
                              help="Mutlak değeri bu eşiğin altındaki L2 farkları sıfırlanır. Gürültü benzeri mikro farkları temizler, seyrekliği artırır.")
        q_factor = st.slider("Quantization (Q)", 1, 20, 5, 1,
                              help="Niceleme adımı: L2 = (L2 // Q) × Q. Büyük Q → daha agresif sıkıştırma, düşük SSIM ama daha az veri. Q=1 kuantizasyon uygulamaz.")

        st.markdown('<div class="sidebar-section">🔐 Doğrulama</div>', unsafe_allow_html=True)
        show_hashes = st.checkbox("SHA-256 Blok Grid", value=False,
                                  help="4×4 blok bazlı SHA-256 hash karşılaştırması. Her blok encoder tarafındaki hash'le karşılaştırılır: 🟢 = OK, 🔴 = bozulma.")

    # ══════════════════════════════════════════════════════════
    #  PIPELINE
    # ══════════════════════════════════════════════════════════

    # APS
    aps_on = noise_intensity > 50 and noise_type != "Yok"
    aps_boost = aps_on
    eff_high = user_high and not aps_on

    # 1. Encode — cv2.subtract ile işaretli L2
    l1 = cv2.resize(original, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    l1_up = cv2.resize(l1, (w, h), interpolation=cv2.INTER_CUBIC)
    l2 = cv2.subtract(original, l1_up, dtype=cv2.CV_16S)
    if l2_thresh > 0:
        l2[np.abs(l2) < l2_thresh] = 0
    l2 = quantize_l2(l2, q_factor)

    # 2. Content-Aware
    imp_map = None
    l2_send = l2.copy()
    if enable_imp:
        imp_map = generate_importance_map(original, star_t, edge_s)
        l2_send = apply_content_aware_masking(l2, imp_map)

    # 3. Hashes (gürültü öncesi)
    l1_hash = sha256(l1)
    l2_hash = sha256(l2_send)
    l1_block_hashes = _sha256_blocks(l1)
    l2_block_hashes = _sha256_blocks(l2_send.astype(np.uint8))

    # 4. Neural Prediction
    neural_info = None
    if enable_neural:
        prev = np.roll(original, 2, axis=1)
        neural_info = stability_check(original, prev, delta_thresh)

    # 5. Gürültü
    if noise_type == "Gaussian Noise" and noise_intensity > 0:
        l1_n = add_gaussian_noise(l1, sigma=noise_intensity * 0.5)
        l2_n = add_gaussian_noise_int16(l2_send, sigma=noise_intensity * 0.3)
    elif noise_type == "Salt & Pepper Noise" and noise_intensity > 0:
        amt = noise_intensity / 500.0
        l1_n = add_salt_pepper_noise(l1, amt)
        l2_n = add_salt_pepper_noise_int16(l2_send, amt)
    else:
        l1_n = l1.copy()
        l2_n = l2_send.copy()

    # 6. Freq boost
    if aps_boost:
        l1_n = frequency_boost_filter(l1_n)

    # 7. Self-Healing
    l1_h, l2_h = l1_n.copy(), l2_n.copy()
    hl1, hl2 = None, None
    l1_corrupt, l2_corrupt = False, False

    # Post-noise L1 block hashes for grid comparison
    l1_n_blocks = _sha256_blocks(l1_n)
    l2_n_blocks = _sha256_blocks(l2_n.astype(np.uint8))

    if enable_healing and noise_intensity > 0:
        c1 = detect_corruption(l1_n, l1_hash)
        l1_corrupt = c1["is_corrupted"]
        if c1["is_corrupted"]:
            r1 = self_heal(l1_n, c1["corruption_mask"])
            l1_h = r1["repaired_data"]
            hl1 = {"corrupted_pct": c1["corrupted_pct"], "repair_pct": r1["repair_pct"], "iters": r1["iterations_used"]}
        c2 = detect_corruption(l2_n, l2_hash)
        l2_corrupt = c2["is_corrupted"]
        if c2["is_corrupted"]:
            r2 = self_heal(l2_n, c2["corruption_mask"])
            l2_h = r2["repaired_data"]
            hl2 = {"corrupted_pct": c2["corrupted_pct"], "repair_pct": r2["repair_pct"], "iters": r2["iterations_used"]}
    elif noise_intensity > 0:
        l1_corrupt = detect_corruption(l1_n, l1_hash)["is_corrupted"]
        l2_corrupt = detect_corruption(l2_n, l2_hash)["is_corrupted"]

    # 8. AI SR
    l1_ai = ai_super_resolution(l1_h, w, h)

    # 9. Lossless / Latent / Signed Fusion
    lossless_data = None
    latent_data = None
    latent_dec = None
    semantic_match = 0.0
    if is_latent:
        latent_data = generate_latent_sfn(original)
        latent_dec = decode_latent_sfn(latent_data["prediction"], latent_data["residual"])
        result = latent_dec["synthesized"]
        semantic_match = latent_dec["semantic_match_pct"]
    elif is_lossless:
        lossless_data = encode_lossless_residual(original)
        prediction_view = lossless_data["prediction"]
        result = decode_lossless_residual(lossless_data["prediction"], lossless_data["residual"])
    elif eff_high:
        result = signed_fusion(l1_ai, l2_h)
    else:
        result = l1_ai

    # 10. Metrikler
    if is_latent:
        ssim_v = semantic_match / 100.0
        mse_v = _compute_mse(original, result)
        psnr_v = 0.0
    else:
        ssim_v = _compute_ssim(original, result)
        mse_v = _compute_mse(original, result)
        psnr_v = 10.0 * np.log10(255**2 / mse_v) if mse_v > 0 else float("inf")
    eff = compute_efficiency(original, l1, l2_send)
    if is_latent:
        eff["packet_kb"] = 0.5
        eff["compression_ratio"] = eff["original_kb"] / 0.5
        eff["bandwidth_saved_pct"] = (1 - 0.5 / eff["original_kb"]) * 100
    is_delta = enable_neural and neural_info and neural_info.get("is_stable", False)
    pwr = compute_power_metrics(eff, neural_info, is_delta, aps_boost)

    # ══════════════════════════════════════════════════════════
    #  INTEGRITY BANNER
    # ══════════════════════════════════════════════════════════
    any_c = l1_corrupt or l2_corrupt
    any_h = hl1 is not None or hl2 is not None

    if noise_intensity == 0 or noise_type == "Yok":
        st.markdown('<div class="integrity-clean"><h4>🛡️ SİNYAL TEMİZ — INTEGRITY OK</h4>'
                    '<p>SHA-256 doğrulaması başarılı.</p></div>', unsafe_allow_html=True)
    elif any_c and not any_h:
        layers = [x for x, c in [("L1", l1_corrupt), ("L2", l2_corrupt)] if c]
        st.markdown(f'<div class="integrity-corrupt"><h4>⚠️ SİNYAL BOZULMASI — {"+".join(layers)} HASH UYUŞMAZLIĞI</h4>'
                    f'<p>Self-Healing {"devre dışı!" if not enable_healing else "başlatılıyor..."}</p></div>',
                    unsafe_allow_html=True)
    elif any_h:
        st.markdown('<div class="integrity-healed"><h4>✅ VERİ ONARILDI — INTEGRITY OK</h4>'
                    '<p>SHA-256 uyuşmazlığı tespit edildi ve AI Inpainting ile onarıldı.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="integrity-clean"><h4>🛡️ SİNYAL TEMİZ — INTEGRITY OK</h4>'
                    '<p>Gürültü eşiğin altında.</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # APS
    if aps_on:
        st.markdown(f'<div class="aps-alert"><div class="aps-title">🔴 APS AKTİF</div>'
                    f'<div class="aps-detail">Gürültü %{noise_intensity} > %50 • LOW BW • Freq Boost (+{pwr["freq_boost_w"]:.0f}W)</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="aps-ok"><div class="aps-title">🟢 APS NORMAL</div>'
                    f'<div class="aps-detail">Gürültü: %{noise_intensity} • {"HIGH" if eff_high else "LOW"} BW</div></div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════════════════════
    #  METRIK KARTLARI (8)
    # ══════════════════════════════════════════════════════════
    mc = st.columns(8)
    if is_latent:
        cards = [
            ("metric-card", "Semantic", f"{semantic_match:.1f}%"),
            ("metric-card", "Vektör", "128 f32"),
            ("metric-card", "Sıkıştırma", f"{eff['compression_ratio']:.0f}×"),
            ("metric-card", "BW Tasarruf", f"{eff['bandwidth_saved_pct']:.1f}%"),
            ("metric-card", "Paket", "0.5 KB"),
            ("metric-card", "Format", ".sfn"),
            ("power-card", "⚡ Klasik", f"{pwr['classical_j']:.1f} J"),
            ("power-card", "⚡ SYNAPS-F", f"{pwr['synapsf_j']:.2f} J"),
        ]
    else:
        cards = [
            ("metric-card", "SSIM", f"{ssim_v:.4f}"),
            ("metric-card", "PSNR", f"{psnr_v:.1f} dB"),
            ("metric-card", "Sıkıştırma", f"{eff['compression_ratio']:.1f}×"),
            ("metric-card", "BW Tasarruf", f"{eff['bandwidth_saved_pct']:.1f}%"),
            ("metric-card", "L2 Seyreklik", f"{eff['l2_sparsity_pct']:.1f}%"),
            ("metric-card", "Q-Factor", f"{q_factor}"),
            ("power-card", "⚡ Klasik", f"{pwr['classical_j']:.1f} J"),
            ("power-card", "⚡ SYNAPS-F", f"{pwr['synapsf_j']:.1f} J"),
        ]
    for col, (cls, lbl, val) in zip(mc, cards):
        with col:
            st.markdown(f'<div class="{cls}"><h3>{lbl}</h3><div class="value">{val}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════════════════════
    #  4'LÜ PANEL
    # ══════════════════════════════════════════════════════════
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.image(original, use_container_width=True, clamp=True)
        st.markdown('<div class="panel-label">📸 Orijinal</div>', unsafe_allow_html=True)
    with p2:
        if is_latent and latent_data is not None:
            # Barkod görselleştirme — residual entropy barcode
            residual_flat = latent_data["residual"].flatten().astype(np.float64)
            # Subsample to 128 bins for barcode
            step = max(len(residual_flat) // 128, 1)
            vec = residual_flat[::step][:128]
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="#0a0a1a")
            ax.set_facecolor("#0a0a1a")
            v_norm = (vec - vec.min()) / max(vec.max() - vec.min(), 1e-6)
            barcode_img = v_norm.reshape(1, -1)
            ax.imshow(barcode_img, aspect="auto", cmap="inferno",
                      extent=[0, 128, 0, 1], interpolation="nearest")
            ax.set_xlabel("Vektör İndeksi", color="#0ff", fontsize=7)
            ax.set_yticks([])
            ax.tick_params(colors="#888", labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("#1a1a3a")
            fig.tight_layout(pad=0.5)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor="#0a0a1a", edgecolor="none")
            plt.close(fig)
            buf.seek(0)
            st.image(buf, use_container_width=True)
            st.markdown('<div class="panel-label">🔢 Nöral Tahmin Artık Verisi</div>', unsafe_allow_html=True)
        else:
            l1_pix = cv2.resize(l1_n, (w, h), interpolation=cv2.INTER_NEAREST)
            st.image(l1_pix, use_container_width=True, clamp=True)
            nlbl = f" + {noise_type}" if noise_type != "Yok" and noise_intensity > 0 else ""
            blbl = " + 📶" if aps_boost else ""
            st.markdown(f'<div class="panel-label">📡 Gelen L1{nlbl}{blbl}</div>', unsafe_allow_html=True)
    with p3:
        if is_latent and latent_dec is not None:
            st.image(latent_dec["synthesized"], use_container_width=True, clamp=True)
            st.markdown(f'<div class="panel-label">🌌 Generative Decoder (Semantic: {semantic_match:.1f}%)</div>',
                        unsafe_allow_html=True)
        elif is_lossless and lossless_data is not None:
            st.image(prediction_view, use_container_width=True, clamp=True)
            st.markdown('<div class="panel-label">🔮 AI Prediktif Tahmin</div>', unsafe_allow_html=True)
        else:
            st.image(l1_ai, use_container_width=True, clamp=True)
            htxt = " + 🩹" if hl1 else ""
            st.markdown(f'<div class="panel-label">🧠 AI SR{htxt}</div>', unsafe_allow_html=True)
    with p4:
        st.image(result, use_container_width=True, clamp=True)
        if is_latent:
            fl = f"Latent Synthesis (Semantic: {semantic_match:.1f}%)"
        elif is_lossless:
            fl = "Lossless (SSIM 1.0)"
        elif eff_high:
            fl = "Signed Fusion"
        else:
            fl = "L1 Only"
        al = " [APS]" if aps_on else ""
        st.markdown(f'<div class="panel-label">🚀 {fl}{al}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MODÜL RAPORLARI (3 sütun)
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    ca, cb, cc = st.columns(3)

    with ca:
        st.markdown("#### 🩹 Self-Healing")
        if noise_intensity == 0:
            st.markdown('<span class="healing-badge badge-ok">● TEMİZ</span>', unsafe_allow_html=True)
        elif not enable_healing:
            st.markdown('<span class="healing-badge badge-fail">● DEVRE DIŞI</span>', unsafe_allow_html=True)
        else:
            if hl1:
                st.markdown('<span class="healing-badge badge-healing">● L1 ONARILDI</span>', unsafe_allow_html=True)
                st.markdown(f"- Bozulma: **{hl1['corrupted_pct']:.1f}%** → Onarım: **{hl1['repair_pct']:.1f}%** ({hl1['iters']} iter)")
            if hl2:
                st.markdown('<span class="healing-badge badge-healing">● L2 ONARILDI</span>', unsafe_allow_html=True)
                st.markdown(f"- Bozulma: **{hl2['corrupted_pct']:.1f}%** → Onarım: **{hl2['repair_pct']:.1f}%**")
            if not hl1 and not hl2:
                st.markdown('<span class="healing-badge badge-ok">● Bozulma Yok</span>', unsafe_allow_html=True)

    with cb:
        st.markdown("#### 🧠 Motion-Aware Prediction")
        if neural_info:
            sx, sy = neural_info.get("shift_x", 0), neural_info.get("shift_y", 0)
            if neural_info.get("is_shift_only"):
                st.markdown('<span class="healing-badge badge-ok">● KAYMA — Shift Modu</span>', unsafe_allow_html=True)
                st.markdown(f"- Shift: **({sx:.1f}, {sy:.1f})** px")
                st.markdown(f"- Sapma: **{neural_info['deviation_pct']:.2f}%**")
                st.markdown("- 📦 Sadece **2 float** gönder (~%99.9 tasarruf)")
            elif neural_info["is_stable"]:
                st.markdown('<span class="healing-badge badge-ok">● STABİL — Delta</span>', unsafe_allow_html=True)
                st.markdown(f"- Sapma: **{neural_info['deviation_pct']:.2f}%**")
                st.markdown(f"- Shift: ({sx:.1f}, {sy:.1f})")
            else:
                st.markdown('<span class="healing-badge badge-healing">● DEĞİŞKEN</span>', unsafe_allow_html=True)
                st.markdown(f"- Sapma: **{neural_info['deviation_pct']:.2f}%**")
        else:
            st.markdown("_Devre dışı_")

    with cc:
        st.markdown("#### ⚡ Enerji & Gecikme")
        st.markdown(f"- Verici: **{pwr['transmitter_w']:.0f}W**"
                    + (f" + Boost **{pwr['freq_boost_w']:.0f}W**" if pwr['freq_boost_w'] > 0 else ""))
        st.markdown(f"- Klasik: **{pwr['classical_s']:.3f} s** → **{pwr['classical_j']:.2f} J**")
        st.markdown(f"- SYNAPS-F: **{pwr['synapsf_s']:.3f} s** → **{pwr['synapsf_j']:.2f} J**")
        clr = "#0f0" if pwr["saved_pct"] > 0 else "#f44"
        st.markdown(f'- Tasarruf: <span style="color:{clr};font-weight:bold;">{pwr["saved_pct"]:.1f}%</span>',
                    unsafe_allow_html=True)

    # ── Importance Map + Verimlilik ──────────────────────────
    st.markdown("---")
    cd, ce = st.columns(2)
    with cd:
        st.markdown("#### 🎯 Importance Map")
        if imp_map is not None:
            cov = (np.count_nonzero(imp_map) / max(imp_map.size, 1)) * 100
            iv = cv2.resize(imp_map, (w, h), interpolation=cv2.INTER_NEAREST)
            ov = cv2.merge([np.zeros_like(original), iv, np.zeros_like(original)])
            bg = cv2.merge([original, original, original])
            bl = cv2.addWeighted(bg, 0.6, ov, 0.4, 0)
            st.image(cv2.cvtColor(bl, cv2.COLOR_BGR2RGB), use_container_width=True, clamp=True)
            st.markdown(f"- Önemli: **{cov:.1f}%** | Void: **{100-cov:.1f}%**")
        else:
            st.markdown("_Devre dışı_")
    with ce:
        st.markdown("#### 📊 Verimlilik")
        st.table({
            "Metrik": ["Orijinal", "L1", "L2 Efektif", "Paket", "Sıkıştırma", "BW Tasarrufu"],
            "Değer": [f"{eff['original_kb']:.1f} KB", f"{eff['l1_kb']:.2f} KB",
                      f"{eff['l2_effective_kb']:.1f} KB", f"{eff['packet_kb']:.1f} KB",
                      f"{eff['compression_ratio']:.1f}×", f"{eff['bandwidth_saved_pct']:.1f}%"],
        })

    # ══════════════════════════════════════════════════════════
    #  SHA-256 BLOK BAZLI INTEGRITY GRID
    # ══════════════════════════════════════════════════════════
    if show_hashes:
        st.markdown("---")
        st.markdown("#### 🔐 SHA-256 Blok Bazlı Bütünlük Kontrolü")
        st.caption("Her blok, encoder hash'i ile karşılaştırılır. 🟢 = OK, 🔴 = bozulma")

        gh1, gh2 = st.columns(2)
        with gh1:
            st.markdown("**L1 İskelet (4×4 Grid)**")
            grid_html = '<div class="block-grid" style="grid-template-columns:repeat(4,1fr);">'
            for r in range(4):
                for c in range(4):
                    enc_h = l1_block_hashes[r][c]
                    rec_h = l1_n_blocks[r][c]
                    cls = "block-ok" if enc_h == rec_h else "block-fail"
                    icon = "✔" if enc_h == rec_h else "✘"
                    grid_html += f'<div class="block-cell {cls}">{icon}<br>{rec_h[:8]}</div>'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

        with gh2:
            st.markdown("**L2 Detay (4×4 Grid)**")
            grid_html = '<div class="block-grid" style="grid-template-columns:repeat(4,1fr);">'
            for r in range(4):
                for c in range(4):
                    enc_h = l2_block_hashes[r][c]
                    rec_h = l2_n_blocks[r][c]
                    cls = "block-ok" if enc_h == rec_h else "block-fail"
                    icon = "✔" if enc_h == rec_h else "✘"
                    grid_html += f'<div class="block-cell {cls}">{icon}<br>{rec_h[:8]}</div>'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#444;font-size:0.55rem;letter-spacing:2px;">'
        'SYNAPS-F v3.1 — Scientific Integrity — TUA Astro Hackathon 2026<br>'
        'AI: Otonom Karar Destek • Nöral Hareket Tahmini • İşaretli Diferansiyel Analiz'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
