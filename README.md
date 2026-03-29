# tuaGrupProjesi

## Projenin Özü ve Vizyonu
Uzay görevlerinde en kısıtlı ve en pahalı kaynak yakıt değil, **bant genişliğidir**. Klasik uzay iletişiminde uydular çektikleri devasa fotoğrafları piksel piksel Dünya'ya göndermeye çalışır. Bu süreç yavaştır, frekansları tıkar ve derin uzay görevlerinde büyük gecikmelere (latency) yol açar.

**SYNAPS-F Yaklaşımı:** *"Görüntünün tamamını gönderme, sadece yapay zekanın Dünya'da tahmin edemeyeceği matematiksel farkları gönder."*

Sistemimiz, uydudaki kısıtlı işlemci gücüyle (Edge Computing) görüntünün ana iskeletini çıkarır, detayları hesaplar ve Dünya'ya ultra sıkıştırılmış, çok küçük bir paket yollar. Dünya'daki yer istasyonunda (Master Node) bulunan güçlü yapay zeka ise bu ufak ipuçlarını alıp görüntünün orijinalini %100'e yakın bir doğrulukla yeniden inşa eder.

---

## Katmanlı Veri Mimarisi
Görüntüyü tek bir dosya olarak değil, iletişim bant genişliğini optimize etmek için mantıksal katmanlara bölerek işliyoruz:

* **L1 (Semantik İskelet - Layer 1):** Orijinal görüntünün yaklaşık 1/8 oranında küçültülmüş, aşırı hafifletilmiş halidir. Fotoğraftaki objelerin nerede olduğunu, genel renk ve ışık dağılımını (dağların silüeti, gezegenin ufuk çizgisi) taşır. Uyduda oluşturulur.
* **L2 (Artık Fark / Neural Residual - Layer 2):** L1 iskeletinde bulunmayan ince detayların (krater uçları, yıldızlar) haritasıdır. Orijinal fotoğraftan L1'in matematiksel olarak çıkarılmasıyla elde edilir. Çoğunlukla sıfırlardan oluşan, aşırı sıkıştırılabilir bir veri setidir. Uyduda oluşturulur.
* **L3 (Tam Rekonstrüksiyon - Layer 3):** Dünya'da (Yer İstasyonunda) oluşur. Yapay zeka L1'i alır, AI Super Resolution ile pürüzsüzleştirir ve uzaydan gelen L2 ince detay haritasını **Signed Fusion** (işaretli birleştirme) metoduyla üzerine dikerek yüksek çözünürlüklü final görüntüsünü yaratır.

---

## En Agresif Sıkıştırma: `.sfn` Formatı
Mod 3'te kullandığımız **SYNAPS-F Network (.sfn)** formatı, sıfır entropi hedefleyen bilimsel bir sıkıştırma modudur.

1. **Nedensel Tahmin (Causal Predictor):** Klasik pikseller yerine algoritma her pikseli, kendi solundaki ve üstündeki piksele bakarak tahmin eder. 
2. **Fark Kodlama:** Eğer tahmin doğruysa piksel iletilmez (0 yazılır). Sadece yanıldığımız piksellerin farkı kaydedilir.
3. **ZLIB Entropy Coding:** Elde edilen verideki Shannon Entropisi dibe vurduğu an, ZLIB kodlaması devreye girerek kilobaytlarca veriyi sadece birkaç yüz baytlık (örn: ~0.5 KB) bir paket haline getirir.

---

## Sistem Akışı: Uydudan Yer İstasyonuna

### Uzay (Edge) Tarafı:
1. **Veri Girişi ve Ön İşleme:** Sensörden gelen görüntü tek kanallı gri tonlamalı (Grayscale) formata (NASA NavCam standardı) çevrilir.
2. **AI Analizi ve Önem Haritası:** `generate_importance_map` ile fotoğraftaki yıldızlar ve krater/kaya sınırları (Canny Edge) tespit edilerek veriler "semantik olarak önemli/önemsiz" diye ayrılır.
3. **Sıkıştırma (Quantization & Entropy):** İnsan gözünün/bilimin umursamayacağı ışık sapmaları Q-Faktör ile sıfırlanır, çıkan L1 ve L2 katmanları sıkıştırılır.
4. **Güvenli İletim:** Veri paketlerine SHA-256 şifreleme ile "hash" (parmak izi) eklenir ve uzay boşluğuna (simüle edilmiş radyasyon/gürültü kanalı) aktarılır.

### Dünya (Master Node) Tarafı:
5. **Doğrulama ve Self-Healing (Öz-Onarım):** Dünya'daki sunucu paketi alır ve SHA-256 hash kontrolü yapar. Yolda radyasyon/gürültü veriyi bozmuşsa AI Inpainting modülü sağlam pikselleri referans alarak bozuk kısımları onarır.
6. **Katmanlı Veri Açılımı:** Temizlenen L1 iskeleti büyütülür, onarılan L2 ince detayları L1'in üzerine eklenir ve L3 (orijinal kalitede) görüntü devasa bant genişliği tasarrufuyla elde edilir.

---

## Performans Metrikleri
Başarımızı kanıtlamak için 3 evrensel NASA/ESA standardı kullanıyoruz:

* **Bant Genişliği Tasarrufu (Bandwidth Saved):** %85 - %99 oranında iletişim ağı rahatlaması.
* **SSIM (Structural Similarity Index):** Taktiksel görevlerde **0.95+**, bilimsel görevlerde **0.99+** (İnsan gözünün ve AI modellerinin algıladığı yapısal benzerlik).
* **PSNR (Peak Signal-to-Noise Ratio):** "Kayıpsız (Lossless/Pure Science)" modumuzda Sonsuz (**99.9 dB**), yani sıfır matematiksel sapma.

---

## Vizyon ve Kullanım Alanları
Bu proje sadece bir hackathon fikri değil, gerçek dünyanın iletişim darboğazlarını çözen bir üründür:

* **Mars ve Derin Uzay Araştırmaları:** Mars Rover'larından gelen yüksek boyutlu NavCam/Mastcam verilerinin gecikmesiz Dünya'ya iletilmesi.
* **Yeryüzü Gözlem Uyduları (LEO):** Göktürk ve İMECE gibi uyduların, kısıtlı yer istasyonu geçiş sürelerinde (pass time) maksimum veriyi yere indirebilmesi.
* **Kayıpsız Bilimsel Veri İletimi:** James Webb gibi teleskoplar için PURE SCIENCE modunda hiçbir piksel sapması olmadan bant tasarrufu yapılması.
* **Kriz ve Taktiksel İletişim:** İletişim altyapısının çöktüğü afet bölgelerinde veya elektronik harp (jamming) altındaki İHA/SİHA operasyonlarında kesintisiz görsel aktarım.
