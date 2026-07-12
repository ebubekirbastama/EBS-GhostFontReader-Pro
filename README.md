# GhostFont CNN Live Analyzer

Basit kullanım kılavuzu.

## Gereksinimler

```bash
pip install -r requirements.txt
```

---

## 1. GUI ile Kullanım

Programı çalıştırın:

```bash
python gui.py
```

Adımlar:

1. **Video** seçin.
2. **Çıktı klasörünü** seçin.
3. İsterseniz **Tesseract OCR** seçeneğini işaretleyin.
4. **Alan Seç ve Gelişmiş Analiz** butonuna basın.
5. İlk kare açıldığında yalnızca hareketli yazının bulunduğu alanı seçin.
6. Analiz tamamlandıktan sonra sonuçlar otomatik olarak çıkış klasörüne kaydedilir.

---

## 2. Komut Satırı

Temel kullanım:

```bash
python ghost_reader.py video.webm
```

OCR ile:

```bash
python ghost_reader.py video.webm --ocr
```

Özel çıktı klasörü:

```bash
python ghost_reader.py video.webm -o ghost_output
```

ROI kullanımı:

```bash
python ghost_reader.py video.webm --roi 120,180,600,220
```

---

## 3. Tampermonkey Kullanımı

`ghost_font_live_analyzer.user.js` dosyasını Tampermonkey'e yükleyin.

Ardından:

1. https://www.mixfont.com/ghost-font adresini açın.
2. Sayfa yüklendiğinde analiz paneli otomatik açılır.
3. Hareket analizi gerçek zamanlı olarak yapılır.

Panel seçenekleri:

- Duraklat
- Sıfırla
- Ters Çevir
- Hareket Enerjisi
- Kare Farkı
- Zamansal Aralık
- Birleşik Analiz

---

## Üretilen Dosyalar

```
12_BEST_READABLE.png
13_BEST_BINARY.png
14_BEST_BINARY_INVERTED.png
15_comparison_montage.png
report.json
vertical_motion.csv
```

