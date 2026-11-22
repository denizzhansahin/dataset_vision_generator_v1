# YOLO26 Çift Modlu (Object Detection & Segmentation) Veri Seti ve Eğitim Kılavuzu

Bu sistem, **MakeSense.ai** üzerinden alınan Object Detection (`.zip` / `.txt`) ve Instance Segmentation (`.json` COCO)[cite: 1] etiketlerini birbirine karıştırmadan, **etiketsiz görselleri veri setine kesinlikle dahil etmeden** ve **görsellerin orijinal piksel çözünürlüklerine (`1920x1080` / `1080x1920`)[cite: 1] hiç dokunmadan** sıfırdan YOLO veri setine dönüştürür.

---

## 1. Temel Prensipler

1. **Sıfır Etiketsiz Görsel:** Kaynak klasörlerdeki etiketsiz görseller veri setine kopyalanmaz ve asla boş `.txt` dosyası oluşturulmaz. Ayrıca bir görselde sadece segmentasyon etiketi varsa ve Object Detection etiketi yoksa, `2_mod_degistir.py` çalıştırıldığında ilgili modda etiketi olmayan görseller otomatik olarak `images_pasif/` klasörüne park edilir.
2. **Orijinal Çözünürlük Korunur:** Görseller kopyalanırken veya 5'li veri çoğaltma (augmentation) uygulanırken piksel boyutları (`width` x `height`) değiştirilmez.
3. **Etiketler Karışmaz:** `.zip` içindeki 5 değerlik kutu etiketleri `labels_obj/` klasörüne, `.json` içindeki poligon koordinatları[cite: 1] `labels_segm/` klasörüne yazılır.

---

## 2. Desteklenen Kaynak Klasör Yapıları

### Senaryo A: Çoklu Grup Klasörlü Yapı
```text
proje_klasoru/
├── augmentation_module.py
├── 1_dataset_olustur.py
├── 2_mod_degistir.py
├── class.txt                     # mahonya, snail
├── class_segm.txt                # diseased_leaf, healthy_leaf
└── zaman_damgali_frameler/
    ├── Grup1/
    │   ├── *.png                 # Görseller
    │   ├── grup1_obj.zip         # MakeSense Object Detection ZIP çıktısı
    │   └── grup1_segm.json       # MakeSense Segmentation COCO JSON çıktısı
    ├── grup2/
    │   ├── *.png
    │   ├── grup2_obj.zip
    │   └── grup2_segm.json
    └── klasor_1/
        └── ...
```

### Senaryo B: Tek Klasörlü (Birleştirilmiş) Düz Yapı
Tüm görseller, `.txt` dosyaları (MakeSense çıktısı), `.json` (COCO poligon çıktısı) ve `.zip` dosyaları doğrudan tek bir klasör altında (`zaman_damgali_frameler/`) toplansa bile sistem otomatik olarak çalışır:

```text
proje_klasoru/
├── augmentation_module.py
├── 1_dataset_olustur.py
├── 2_mod_degistir.py
├── class.txt                     # insan (veya mahonya, snail)
├── class_segm.txt                # diseased_leaf, healthy_leaf
└── zaman_damgali_frameler/
    ├── 0-00-00.101İnsan.png      # Görseller
    ├── 0-00-00.101İnsan.txt      # Açık .txt etiketleri (5 sütun: obj, >=7 sütun: segm)
    ├── 0-00-00.202İnsan.png
    ├── 0-00-00.202İnsan.txt
    ├── labels_my-project.../     # (İsteğe bağlı) MakeSense'den çıkarılan etiket alt klasörü
    ├── etiketler.zip             # (İsteğe bağlı) MakeSense Detection ZIP çıktısı
    ├── segmentasyon.json         # (İsteğe bağlı) MakeSense COCO JSON çıktısı
    └── class.txt                 # (İsteğe bağlı) Klasöre özel sınıf dosyası
```

* **Unicode & Türkçe Karakter Desteği:** macOS (NFD) ve Windows/Linux (NFC) Unicode farkları (`İ` harfi, `ı`, `ö`, `ü`, `ç`, `ş`, `ğ`) otomatik normalize edilir; görsel adı ile `.txt` dosya adları kusursuz eşleşir.
* **Otomatik Ayrıştırma:**
  - **5 sütunlu `.txt` satırları ve `.zip`** -> `labels_obj/` (Object Detection)
  - **>=7 sütunlu `.txt` satırları ve COCO `.json`** -> `labels_segm/` (Instance Segmentation)
* **Alt Klasörlü Etiket Desteği:** `.txt`, `.zip` veya `.json` dosyaları `zaman_damgali_frameler` içinde bir alt klasörde olsa dahi otomatik taranır.

---

## 3. Adım Adım Çalıştırma Rehberi

### Adım 1: Kütüphanelerin Kurulumu
```bash
pip install opencv-python numpy ultralytics
```

### Adım 2: Veri Setini Sıfırdan Oluşturma
```bash
python 1_dataset_olustur.py
```
* **Oran Seçimi:** Varsayılan olarak **%70 Train, %20 Val, %10 Test** oranında ayırır (İsteğe bağlı değiştirilebilir).
* **Veri Çoğaltma (Augmentation):** Kullanıcı onay verirse `train` setindeki görsellere 5 farklı alanda (Yatay Çevirme, Dikey Çevirme, Parlaklık/Kontrast, HSV Renk Kaydırma, Gürültü/Bulanıklık) çoğaltma uygular ve koordinatları otomatik hesaplar.
* **İstatistik Raporu:** Taranan, atlanan (etiketsiz) ve kabul edilen görsel sayılarını; `labels_obj` (`mahonya`, `snail`[cite: 3]) ve `labels_segm` (`diseased_leaf`, `healthy_leaf`[cite: 1, 2]) için `.txt` dosya sayılarını ve sınıf başına düşen toplam etiket sayılarını ekrana yazdırır.

### Adım 3: Eğitim Öncesi Mod Seçimi
```bash
python 2_mod_degistir.py
```
* **`1` (Object Detection):** `labels_obj` klasörünü `labels` yapar, sadece Object Detection etiketi olan görselleri `images/` içinde aktif bırakır ve `data.yaml` dosyasını `class.txt`[cite: 3] ile günceller.
* **`2` (Instance Segmentation):** `labels_segm` klasörünü `labels` yapar, sadece Segmentation etiketi olan görselleri `images/` içinde aktif bırakır ve `data.yaml` dosyasını `class_segm.txt`[cite: 2] ile günceller.

---

## 4. YOLO26 Eğitim Komutları

### Object Detection Eğitimi (`2_mod_degistir.py` -> `1` seçildikten sonra)
```bash
yolo detect train data=dataset/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=16
```

### Instance Segmentation Eğitimi (`2_mod_degistir.py` -> `2` seçildikten sonra)
```bash
yolo segment train data=dataset/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640 batch=16
```

### Test Veri Seti (%10) Üzerinde Doğruluk Ölçümü
```bash
yolo val data=dataset/data.yaml model=runs/segment/train/weights/best.pt split=test
```