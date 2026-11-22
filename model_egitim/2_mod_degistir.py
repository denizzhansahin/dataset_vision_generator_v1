import unicodedata
from pathlib import Path

SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def normalize_stem(stem_str):
    if not stem_str:
        return ""
    return unicodedata.normalize("NFC", str(stem_str)).strip()


def read_classes(file_path, default_classes, search_dirs=None):
    candidates = [Path(file_path)]
    if search_dirs:
        for sd in search_dirs:
            if sd:
                p = Path(sd)
                candidates.append(p / Path(file_path).name)
                candidates.append(p / "class.txt")
                candidates.append(p / "classes.txt")
    candidates.extend([
        Path(".") / Path(file_path).name,
        Path("..") / Path(file_path).name,
        Path(__file__).parent / Path(file_path).name,
        Path(__file__).parent.parent / Path(file_path).name,
    ])

    seen = set()
    for cp in candidates:
        try:
            cp_resolved = cp.resolve()
        except Exception:
            continue
        if cp_resolved in seen:
            continue
        seen.add(cp_resolved)
        if cp.is_file():
            with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if lines:
                return lines
    return default_classes


def clear_yolo_caches(dataset_dir):
    """Önceki eğitimden kalan .cache dosyalarını siler ki etiketler karışmasın."""
    for cache_file in Path(dataset_dir).rglob("*.cache"):
        cache_file.unlink()
        print(f" -> Eski önbellek temizlendi: {cache_file.name}")


def sync_active_images_with_labels(dataset_dir):
    """
    Seçilen modda (labels/) etiketi olmayan görselleri geçici olarak 'images_pasif/'
    klasörüne taşır; etiketi olanları 'images/' içine geri alır.
    Unicode (NFC/NFD) normalizasyonu kullanarak eşleştirme yapar.
    """
    ds_path = Path(dataset_dir)
    images_dir = ds_path / "images"
    pasif_dir = ds_path / "images_pasif"
    labels_dir = ds_path / "labels"

    active_count = 0
    parked_count = 0

    for split in ["train", "val", "test"]:
        split_img_dir = images_dir / split
        split_pasif_dir = pasif_dir / split
        split_lbl_dir = labels_dir / split

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_pasif_dir.mkdir(parents=True, exist_ok=True)

        # 1. Önce pasifte bekleyen tüm görselleri ana images/ klasörüne geri taşı
        if split_pasif_dir.exists():
            for f in split_pasif_dir.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXTS:
                    f.rename(split_img_dir / f.name)

        # 2. Aktif labels/ klasöründeki dolu etiketlerin normalize edilmiş adlarını topla
        valid_label_stems = set()
        if split_lbl_dir.exists():
            for lf in split_lbl_dir.glob("*.txt"):
                if lf.is_file() and lf.stat().st_size > 0:
                    valid_label_stems.add(normalize_stem(lf.stem))

        # 3. Aktif labels/ klasöründe dolu .txt karşılığı olmayan görselleri pasife çek
        for img_file in list(split_img_dir.iterdir()):
            if not (img_file.is_file() and img_file.suffix.lower() in SUPPORTED_IMG_EXTS):
                continue
            norm_stem = normalize_stem(img_file.stem)
            if norm_stem not in valid_label_stems:
                img_file.rename(split_pasif_dir / img_file.name)
                parked_count += 1
            else:
                active_count += 1

    print(f" -> Görsel Senkronizasyonu: {active_count} etiketli görsel aktif, {parked_count} etiketsiz görsel pasife alındı.")


def update_yaml(dataset_dir, classes, mode_label):
    ds_path = Path(dataset_dir).resolve()
    yaml_path = ds_path / "data.yaml"
    names_lines = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])

    yaml_text = f"""# Otomatik Oluşturuldu - Aktif Mod: {mode_label}
path: {ds_path.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_lines}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f" -> 'data.yaml' güncellendi! Aktif Sınıflar: {classes}")


def switch_mode(dataset_dir="./dataset", obj_class_file="./class.txt", segm_class_file="./class_segm.txt"):
    ds_path = Path(dataset_dir)
    labels_active = ds_path / "labels"
    labels_obj = ds_path / "labels_obj"
    labels_segm = ds_path / "labels_segm"

    if not ds_path.exists():
        print(f"Hata: '{ds_path}' bulunamadı! Lütfen önce '1_dataset_olustur.py' betiğini çalıştırın.")
        return

    obj_classes = read_classes(obj_class_file, ["insan"], search_dirs=[ds_path])
    segm_classes = read_classes(segm_class_file, ["diseased_leaf", "healthy_leaf"], search_dirs=[ds_path])

    current_status = "Henüz Seçilmedi ('labels' klasörü yok)"
    if labels_active.exists():
        if not labels_obj.exists() and labels_segm.exists():
            current_status = "OBJECT DETECTION (labels_obj şu an 'labels' olarak aktif)"
        elif not labels_segm.exists() and labels_obj.exists():
            current_status = "SEGMENTATION (labels_segm şu an 'labels' olarak aktif)"

    line_sep = "=" * 62
    print("\n" + line_sep)
    print(f"MEVCUT DURUM: {current_status}")
    print(line_sep)
    print("Hangi YOLO eğitimini başlatmak istiyorsunuz?")
    print(f"  [1] Object Detection      ({', '.join(obj_classes)})")
    print(f"  [2] Instance Segmentation ({', '.join(segm_classes)})")

    choice = input("\nSeçiminiz (1 veya 2): ").strip()

    if choice == "1":
        if labels_active.exists() and not labels_segm.exists():
            labels_active.rename(labels_segm)
            print(" -> Mevcut 'labels' klasörü 'labels_segm' olarak koruma altına alındı.")

        if labels_obj.exists():
            labels_obj.rename(labels_active)
            print(" -> 'labels_obj' klasörü 'labels' olarak aktif edildi.")

        sync_active_images_with_labels(ds_path)
        clear_yolo_caches(ds_path)
        update_yaml(ds_path, obj_classes, "Object Detection")
        print("\n[HAZIR] Object Detection eğitimi için tüm görseller %100 etiketli ve hazır!")

    elif choice == "2":
        if labels_active.exists() and not labels_obj.exists():
            labels_active.rename(labels_obj)
            print(" -> Mevcut 'labels' klasörü 'labels_obj' olarak koruma altına alındı.")

        if labels_segm.exists():
            labels_segm.rename(labels_active)
            print(" -> 'labels_segm' klasörü 'labels' olarak aktif edildi.")

        sync_active_images_with_labels(ds_path)
        clear_yolo_caches(ds_path)
        update_yaml(ds_path, segm_classes, "Instance Segmentation")
        print("\n[HAZIR] Instance Segmentation eğitimi için tüm görseller %100 etiketli ve hazır!")

    else:
        print("Geçersiz seçim! Lütfen 1 veya 2 girin.")


if __name__ == "__main__":
    switch_mode(
        dataset_dir="./dataset",
        obj_class_file="./class.txt",
        segm_class_file="./class_segm.txt"
    )