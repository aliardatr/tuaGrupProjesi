/**
 * ASTRO-LYNX — Mission Control Dashboard
 * Adaptive Semantic Space Data Protocol
 * SYNAPS-F v3.2
 * TUA Astro Hackathon 2026
 */

// ══════════════════════════════════════════════
//  MOCK API DATASETS (Toplu Veri Simülasyonu)
// ══════════════════════════════════════════════
const API_DATASETS = {
    nasa: [
        'datasets/nasa_mars.jpg',
        // Eğer elinde başka nasa fotoğrafları varsa buraya ekleyebilirsin: 'datasets/nasa_mars_2.jpg'
    ],
    tua: [
        'datasets/tua_gokturk.jpg',
        // Eğer elinde başka tua fotoğrafları varsa buraya ekleyebilirsin
    ]
};

// ══════════════════════════════════════════════
//  HEATMAP GENERATOR
// ══════════════════════════════════════════════
class HeatmapGenerator {
    static generate(canvas, originalData, reconstructedData) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const imgData = ctx.createImageData(w, h);

        const scaleX = originalData.width / w;
        const scaleY = originalData.height / h;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const srcX = Math.floor(x * scaleX);
                const srcY = Math.floor(y * scaleY);
                const srcIdx = (srcY * originalData.width + srcX) * 4;

                const dr = Math.abs(originalData.data[srcIdx] - reconstructedData.data[srcIdx]);
                const dg = Math.abs(originalData.data[srcIdx + 1] - reconstructedData.data[srcIdx + 1]);
                const db = Math.abs(originalData.data[srcIdx + 2] - reconstructedData.data[srcIdx + 2]);
                const error = (dr + dg + db) / 3;

                const confidence = Math.max(0, 1 - error / 40);
                const idx = (y * w + x) * 4;

                if (confidence > 0.7) {
                    imgData.data[idx] = Math.round(50 * (1 - confidence));
                    imgData.data[idx + 1] = Math.round(180 + 75 * confidence);
                    imgData.data[idx + 2] = Math.round(80 * confidence);
                } else if (confidence > 0.4) {
                    imgData.data[idx] = Math.round(200 + 55 * (1 - confidence));
                    imgData.data[idx + 1] = Math.round(180 * confidence);
                    imgData.data[idx + 2] = 30;
                } else {
                    imgData.data[idx] = Math.round(200 + 55 * (1 - confidence));
                    imgData.data[idx + 1] = Math.round(50 * confidence);
                    imgData.data[idx + 2] = 30;
                }
                imgData.data[idx + 3] = 200;
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }
}

// ══════════════════════════════════════════════
//  DASHBOARD CONTROLLER
// ══════════════════════════════════════════════
class DashboardController {
    constructor() {
        this.isRunning = false;
        this.originalImage = null; // Base64 veya ImageData
        this.userImage = null;

        // Canvas references
        this.canvasOriginal = document.getElementById('canvas-original');
        this.canvasL1 = document.getElementById('canvas-l1');
        this.canvasL2 = document.getElementById('canvas-l2');
        this.canvasL3 = document.getElementById('canvas-l3');
        this.canvasHeatmap = document.getElementById('canvas-heatmap');

        // API Galeri Container
        this.apiGallery = document.getElementById('api-gallery');

        this.initElements();
        this.bindEvents();
        this.initDynamicUI();
        this._clearProcessingCanvases();
        this._log('ASTRO-LYNX Mission Control başlatıldı.', 'info');
    }

    initElements() {
        this.btnStart = document.getElementById('btn-start');
        this.modeSelector = document.getElementById('mode-selector');
        this.btnApiNasa = document.getElementById('btn-api-nasa');
        this.btnApiTua = document.getElementById('btn-api-tua');
    }

    bindEvents() {
        if (this.btnStart) {
            this.btnStart.addEventListener('click', () => this.startSimulation());
        }

        // API Fetch Eventleri
        if (this.btnApiNasa) {
            this.btnApiNasa.addEventListener('click', () => this.simulateAPIFetch('nasa', 'NASA Mastcam-Z API'));
        }
        if (this.btnApiTua) {
            this.btnApiTua.addEventListener('click', () => this.simulateAPIFetch('tua', 'TUA Göktürk-2 API'));
        }
    }

    initDynamicUI() {
        if (!this.modeSelector) return;

        this.modeSelector.addEventListener('change', (e) => {
            const mode = e.target.value;
            const qFactorGroup = document.getElementById('q-factor-group');

            const l1Label = document.querySelector('#layer-l1 .layer-name');
            const l2Label = document.querySelector('#layer-l2 .layer-name');
            const l1Badge = document.querySelector('.badge-l1');
            const l2Badge = document.querySelector('.badge-l2');

            const simL1Label = l1Badge ? l1Badge.previousElementSibling : null;
            const simL2Label = l2Badge ? l2Badge.previousElementSibling : null;

            if (mode === 'tactical') {
                if (qFactorGroup) qFactorGroup.style.display = 'flex';
                if (l1Label) l1Label.textContent = 'Semantik İskelet';
                if (l2Label) l2Label.textContent = 'Artık Fark (Residual)';
                if (l1Badge) l1Badge.textContent = 'ENCODE';
                if (l2Badge) l2Badge.textContent = 'RESIDUAL';
                if (simL1Label) simL1Label.textContent = 'L1 — Semantik İskelet';
                if (simL2Label) simL2Label.textContent = 'L2 — Artık Fark';
            } else if (mode === 'lossless') {
                if (qFactorGroup) qFactorGroup.style.display = 'none';
                if (l1Label) l1Label.textContent = 'Nöral Tahmin (Prediction)';
                if (l2Label) l2Label.textContent = 'Kayıpsız Artık (Residual)';
                if (l1Badge) l1Badge.textContent = 'AI PRED';
                if (l2Badge) l2Badge.textContent = 'LOSSLESS';
                if (simL1Label) simL1Label.textContent = 'L1 — Nöral Tahmin';
                if (simL2Label) simL2Label.textContent = 'L2 — Kayıpsız Artık';
            } else if (mode === 'latent') {
                if (qFactorGroup) qFactorGroup.style.display = 'none';
                if (l1Label) l1Label.textContent = 'Topolojik Çekirdek';
                if (l2Label) l2Label.textContent = 'İletilmiyor (Sıfır)';
                if (l1Badge) l1Badge.textContent = '11x11 MATRİS';
                if (l2Badge) l2Badge.textContent = 'VOID';
                if (simL1Label) simL1Label.textContent = 'L1 — 1D .sfn Vektör';
                if (simL2Label) simL2Label.textContent = 'L2 — İletilmiyor';
            }
        });
    }

    // ── API Simülasyonu (Jüri Şovu) ──
    simulateAPIFetch(agency, agencyName) {
        if (this.isRunning) return;

        this.apiGallery.innerHTML = '';
        this.apiGallery.style.display = 'grid';
        this._log(`[UPLINK] ${agencyName} ağına güvenli el sıkışma (handshake) başlatıldı...`, 'warn');

        this.btnApiNasa.disabled = true;
        this.btnApiTua.disabled = true;

        document.getElementById('status-text').textContent = 'API BAĞLANILIYOR...';
        document.getElementById('status-dot').className = 'status-dot orange';

        setTimeout(() => {
            this._log(`[UPLINK] Kimlik doğrulandı. Bulk (Toplu) veri paketleri indiriliyor...`, 'info');

            setTimeout(() => {
                const dataset = API_DATASETS[agency];
                this.renderGallery(dataset);
                this._log(`[UPLINK] Yüksek çözünürlüklü veri paketleri başarıyla çekildi. İşlemeye hazır.`, 'success');

                this.btnApiNasa.disabled = false;
                this.btnApiTua.disabled = false;
                document.getElementById('status-text').textContent = 'VERİ HAZIR';
                document.getElementById('status-dot').className = 'status-dot green';
            }, 1200);
        }, 800);
    }

    renderGallery(imagePaths) {
        imagePaths.forEach((path, index) => {
            const img = document.createElement('img');
            img.src = path;
            img.className = 'api-thumb';
            img.alt = `Data Packet ${index + 1}`;

            // Resme tıklandığında ana ekrana yükle
            img.addEventListener('click', () => {
                document.querySelectorAll('.api-thumb').forEach(el => el.classList.remove('active'));
                img.classList.add('active');

                // Resmi canvas'a çiz ve originalImage olarak belirle
                this.loadImageToOriginalCanvas(path, `API_PACKET_${index + 1}`);
            });

            this.apiGallery.appendChild(img);
        });

        // Otomatik olarak ilk resmi seç
        if (this.apiGallery.firstChild) {
            this.apiGallery.firstChild.click();
        }
    }

    // ── Resim Yükleme ve Çizme İşlemleri ──
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this._log('HATA: Lütfen geçerli bir görüntü dosyası yükleyin.', 'warn');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.loadImageToOriginalCanvas(e.target.result, file.name);

            // Preview alanını güncelle
            const previewImg = document.getElementById('preview-img');
            previewImg.src = e.target.result;
            document.getElementById('upload-preview').style.display = 'flex';
            document.getElementById('upload-loaded-text').textContent = `✓ ${file.name} yüklendi`;
            document.getElementById('upload-zone').classList.add('has-image');
            document.querySelector('.upload-title').textContent = file.name;

            // Seçili API resimlerinin aktifliğini kaldır
            document.querySelectorAll('.api-thumb').forEach(el => el.classList.remove('active'));
        };
        reader.readAsDataURL(file);
    }

    loadImageToOriginalCanvas(src, filename) {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 256;
            tempCanvas.height = 256;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(img, 0, 0, 256, 256);

            this.userImage = tempCtx.getImageData(0, 0, 256, 256);
            this.originalImage = this.userImage; // Sistem artık bunu kullanacak

            const ctx = this.canvasOriginal.getContext('2d');
            ctx.putImageData(this.originalImage, 0, 0);

            this._clearProcessingCanvases();
            this._log(`Hedef görüntü kilitlendi: ${filename}`, 'info');
        };
        img.src = src;
    }

    _clearProcessingCanvases() {
        [this.canvasL1, this.canvasL2, this.canvasL3].forEach(canvas => {
            const c = canvas.getContext('2d');
            c.fillStyle = '#0a1628';
            c.fillRect(0, 0, 256, 256);
            c.fillStyle = '#1e293b';
            c.font = '600 12px Orbitron, sans-serif';
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillText('BEKLEMEDE', 128, 120);
        });

        const hCtx = this.canvasHeatmap.getContext('2d');
        hCtx.fillStyle = '#0a1628';
        hCtx.fillRect(0, 0, this.canvasHeatmap.width, this.canvasHeatmap.height);
        hCtx.fillStyle = '#1e293b';
        hCtx.font = '600 14px Orbitron';
        hCtx.textAlign = 'center';
        hCtx.fillText('SİMÜLASYON BEKLENİYOR', this.canvasHeatmap.width / 2, this.canvasHeatmap.height / 2);

        document.getElementById('btn-start').textContent = '▶ SİMÜLASYONU BAŞLAT';
        document.getElementById('btn-start').disabled = false;
        document.getElementById('status-text').textContent = 'SİSTEM HAZIR';
        document.getElementById('status-dot').className = 'status-dot green';

        document.querySelectorAll('.layer-card').forEach(el => el.classList.remove('active'));
        ['l1-progress', 'l2-progress', 'l3-progress'].forEach(id => {
            document.getElementById(id).style.width = '0%';
        });
    }

    // ── ANA SİMÜLASYON (FastAPI'ye Bağlantı) ──
    async startSimulation() {
        if (!this.originalImage) {
            this._log('HATA: Lütfen API menüsünden veya bilgisayardan bir görüntü seçin.', 'warn');
            return;
        }

        if (this.isRunning) return;
        this.isRunning = true;

        document.getElementById('btn-start').textContent = '⏳ İŞLENİYOR...';
        document.getElementById('btn-start').disabled = true;
        document.getElementById('status-text').textContent = 'İŞLENİYOR';
        document.getElementById('status-dot').className = 'status-dot orange';

        const mode = this.modeSelector ? this.modeSelector.value : 'tactical';
        const qFactor = document.getElementById('q-factor-slider') ? document.getElementById('q-factor-slider').value : '5';
        const noise = document.getElementById('noise-slider') ? document.getElementById('noise-slider').value : '0';

        this._log(`[UPLINK] Veri iletiliyor. Mod: ${mode.toUpperCase()} | Q: ${qFactor} | Radyasyon: %${noise}`, 'info');

        try {
            // Base64'e Çevir
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 256;
            tempCanvas.height = 256;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(this.originalImage, 0, 0);
            const base64Image = tempCanvas.toDataURL('image/png');

            document.getElementById('layer-l1').classList.add('active');
            document.getElementById('l1-progress').style.width = '50%';

            const formData = new FormData();
            formData.append('image', base64Image);
            formData.append('mode', mode);
            formData.append('q_factor', qFactor);
            formData.append('noise_intensity', noise);

            // Python Backend API İstek
            const response = await fetch('/process-image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
            const data = await response.json();

            // Canvasları Güncelle (L1, L2, L3)
            await this._drawBase64OnCanvas(data.l1_image, this.canvasL1);
            document.getElementById('l1-progress').style.width = '100%';

            if (data.l2_image) {
                await this._drawBase64OnCanvas(data.l2_image, this.canvasL2);
            }
            document.getElementById('layer-l1').classList.remove('active');
            document.getElementById('layer-l2').classList.add('active');
            document.getElementById('l2-progress').style.width = '100%';

            await this._drawBase64OnCanvas(data.l3_image, this.canvasL3);
            document.getElementById('layer-l2').classList.remove('active');
            document.getElementById('layer-l3').classList.add('active');
            document.getElementById('l3-progress').style.width = '100%';

            // Heatmap Oluştur
            const l3ImageData = this.canvasL3.getContext('2d').getImageData(0, 0, 256, 256);
            HeatmapGenerator.generate(this.canvasHeatmap, this.originalImage, l3ImageData);

            // Arayüz Metriklerini Güncelle
            this._updateGauge('gauge-comp', Math.min(100, (data.metrics.compression_ratio / 20) * 100));
            document.getElementById('gauge-comp-val').textContent = data.metrics.compression_ratio.toFixed(1) + 'x';

            this._updateGauge('gauge-psnr', Math.min(100, (data.metrics.psnr / 50) * 100));
            document.getElementById('gauge-psnr-val').textContent = data.metrics.psnr === 99.9 ? 'LOSSLESS' : data.metrics.psnr.toFixed(1);

            this._updateGauge('gauge-ssim', data.metrics.ssim * 100);
            document.getElementById('gauge-ssim-val').textContent = data.metrics.ssim.toFixed(4);

            this._updateGauge('gauge-bw', Math.max(0, data.metrics.bandwidth_saved_pct));
            document.getElementById('gauge-bw-val').textContent = data.metrics.bandwidth_saved_pct.toFixed(1) + '%';

            // Verify Stats
            this._animateValue('verify-accuracy', 0, data.metrics.ssim * 100, '%', 1000, 1);
            this._animateValue('verify-confidence', 0, data.metrics.confidence * 100, '%', 1000, 1);
            document.getElementById('verify-anomalies').textContent = data.metrics.anomalies;
            document.getElementById('verify-latency').textContent = data.metrics.latency_ms + ' ms';

            // Hero Stats
            this._animateValue('stat-compression', 0, data.metrics.compression_ratio, 'x', 1000, 1);
            this._animateValue('stat-psnr', 0, data.metrics.psnr, ' dB', 1000, 1);
            this._animateValue('stat-ssim', 0, data.metrics.ssim, '', 1000, 4);
            this._animateValue('stat-bandwidth', 0, data.metrics.bandwidth_saved_pct, '%', 1000, 1);

            this._log(`[DOWNLINK] Başarılı! %${data.metrics.bandwidth_saved_pct.toFixed(1)} bant tasarrufu sağlandı.`, 'success');

            if (data.metrics.anomalies > 0) {
                this._log(`[SELF-HEAL] Uzay radyasyonu algılandı. AI Inpainting ile ${data.metrics.anomalies} anomali onarıldı.`, 'warn');
            }

        } catch (error) {
            this._log(`BAĞLANTI HATASI: ${error.message}`, 'warn');
            document.getElementById('status-text').textContent = 'HATA';
            document.getElementById('status-dot').className = 'status-dot red';
        } finally {
            this.isRunning = false;
            document.getElementById('btn-start').textContent = '▶ SİMÜLASYONU BAŞLAT';
            document.getElementById('btn-start').disabled = false;
        }
    }

    _drawBase64OnCanvas(base64Str, canvas) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                resolve();
            };
            img.onerror = () => resolve();
            img.src = 'data:image/png;base64,' + base64Str;
        });
    }

    resetSimulation() {
        this.isRunning = false;
        this.originalImage = null;
        this.userImage = null;

        // Galeriyi ve Preview'i temizle
        this.apiGallery.innerHTML = '';
        document.getElementById('upload-preview').style.display = 'none';
        document.getElementById('upload-zone').classList.remove('has-image');
        document.querySelector('.upload-title').textContent = 'Lokal Dosya Yükle';

        const c = this.canvasOriginal.getContext('2d');
        c.fillStyle = '#0a1628';
        c.fillRect(0, 0, 256, 256);

        this._clearProcessingCanvases();

        // Reset gauges
        ['gauge-psnr', 'gauge-ssim', 'gauge-comp', 'gauge-bw'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.strokeDashoffset = 100;
        });

        document.getElementById('gauge-psnr-val').textContent = '0';
        document.getElementById('gauge-ssim-val').textContent = '0';
        document.getElementById('gauge-comp-val').textContent = '0x';
        document.getElementById('gauge-bw-val').textContent = '0%';

        this._log('Simülasyon paneli sıfırlandı.', 'warn');
    }

    _log(message, type = '') {
        const consoleEl = document.getElementById('log-console');
        if (!consoleEl) return;
        const time = new Date().toISOString().split('T')[1].substring(0, 12);
        const div = document.createElement('div');
        div.className = `log-msg ${type}`;
        div.innerHTML = `<span class="timestamp">[${time}]</span> ${message}`;
        consoleEl.appendChild(div);
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    _updateGauge(elementId, value) {
        const gauge = document.getElementById(elementId);
        if (!gauge) return;
        const circumference = 99.9; // 2 * PI * 15.9
        const offset = circumference - (value / 100) * circumference;
        gauge.style.strokeDashoffset = offset;
    }

    _animateValue(elementId, start, end, suffix, duration, decimals = 1) {
        const element = document.getElementById(elementId);
        if (!element) return;
        const endVal = parseFloat(end);
        const startVal = parseFloat(start);
        const startTime = performance.now();
        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (endVal - startVal) * eased;
            element.textContent = (endVal === 99.9 && elementId === 'stat-psnr') ? 'LOSSLESS' : current.toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(update);
        };
        requestAnimationFrame(update);
    }
}

let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new DashboardController();
});