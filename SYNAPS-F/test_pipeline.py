"""Headless test — Encoder + Decoder full pipeline."""
import matplotlib
matplotlib.use("Agg")          # GUI penceresi açılmasın
import matplotlib.pyplot as plt

# plt.show'u devre dışı bırak (headless ortam)
_original_show = plt.show
plt.show = lambda *a, **kw: None

from synapsf_encoder import synapsf_encoder, _sha256
from synapsf_decoder import synapsf_decoder
import cv2

IMAGE = "test_uzay.jpg"

# Encoder
encoded = synapsf_encoder(IMAGE)
l1 = encoded["l1"]
l2 = encoded["l2"]
original = encoded["original"]
rapor = encoded["hashes"]
print(rapor)

l1_h = _sha256(l1)
l2_h = _sha256(l2)

# Decoder — HIGH BW
dec_h = synapsf_decoder(l1, l2, original, "high", l1_h, l2_h)
res_h, met_h = dec_h["result"], dec_h["metrics"]

# Decoder — LOW BW
dec_l = synapsf_decoder(l1, l2, original, "low", l1_h, l2_h)
res_l, met_l = dec_l["result"], dec_l["metrics"]

print("\n" + "=" * 66)
print("  SONUÇ KARSILASTIRMASI")
print("=" * 66)
print(f"  HIGH BW -> SSIM: {met_h['ssim']:.6f} | MSE: {met_h['mse']:.4f} | PSNR: {met_h['psnr_db']:.2f} dB")
print(f"  LOW  BW -> SSIM: {met_l['ssim']:.6f} | MSE: {met_l['mse']:.4f} | PSNR: {met_l['psnr_db']:.2f} dB")
print("=" * 66)

# Restore
plt.show = _original_show
