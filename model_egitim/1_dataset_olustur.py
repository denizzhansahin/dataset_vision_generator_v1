import json
import zipfile
import shutil
import random
import unicodedata
from pathlib import Path
from collections import Counter
import cv2

from augmentation_module import get_augmentation_config, apply_augmentations_to_item

SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_DEF_NAMES = {"class", "classes", "class_segm", "obj", "labels"}


def normalize_stem(stem_str):
    """
    macOS (APFS/HFS+ NFD) ve Windows/Linux (NFC) Unicode karakter farklılıklarını
    (Örn: 'İ' harfinin 'I' + combining dot üst üste gelmesi) tek standarda eşitler.
    """
    if not stem_str:
        return ""
    return unicodedata.normalize("NFC", str(stem_str)).strip()


def load_class_list(file_path, default_list, search_dirs=None):
    """
    Sınıf listesini belirtilen yoldan veya kaynak klasör, üst klasör vb. alternatiflerden arar.
    """
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
                print(f" -> Sınıf listesi okundu: '{cp}' -> {lines}")
                return lines
    print(f" -> Sınıf dosyası bulunamadı, varsayılan liste kullanılıyor: {default_list}")
    return default_list


def parse_coco_json_to_dict(json_path, segm_class_names=None):
    """
    MakeSense COCO JSON dosyasını okur ve hem poligon segmentasyon hem de
    varsa tespit kutularını normalize edilmiş YOLO formatına dönüştürür.
    Unicode NFD/NFC uyumludur.
    """
    segm_dict = {}
    obj_dict = {}
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception as e:
        print(f" [!] JSON okuma hatası ({json_path.name}): {e}")
        return segm_dict, obj_dict

    if "images" not in data or "annotations" not in data:
        return segm_dict, obj_dict

    categories = sorted(data.get("categories", []), key=lambda x: x["id"])
    cat_id_to_yolo = {}
    for idx, cat in enumerate(categories):
        cat_name = cat.get("name", "").strip()
        if segm_class_names and cat_name in segm_class_names:
            cat_id_to_yolo[cat["id"]] = segm_class_names.index(cat_name)
        else:
            cat_id_to_yolo[cat["id"]] = idx

    images_info = {img["id"]: img for img in data.get("images", [])}

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann.get("image_id")
        if img_id not in images_info:
            continue

        img_data = images_info[img_id]
        img_w, img_h = img_data.get("width", 1), img_data.get("height", 1)
        raw_file_name = img_data.get("file_name", "")
        stem = normalize_stem(Path(raw_file_name).stem)
        yolo_cid = cat_id_to_yolo.get(ann.get("category_id"), 0)

        # 1. Poligon Segmentasyon kontrolü
        has_poly = False
        for poly in ann.get("segmentation", []):
            if len(poly) < 6:
                continue
            norm_coords = []
            for i in range(0, len(poly), 2):
                x = max(0.0, min(1.0, poly[i] / img_w))
                y = max(0.0, min(1.0, poly[i + 1] / img_h))
                norm_coords.append(f"{x:.6f} {y:.6f}")
            segm_dict.setdefault(stem, []).append(f"{yolo_cid} " + " ".join(norm_coords))
            has_poly = True

        # 2. Eğer segmentasyon poligonu yoksa ama bbox varsa Object Detection olarak al
        if not has_poly and "bbox" in ann:
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                xc = max(0.0, min(1.0, (bx + bw / 2.0) / img_w))
                yc = max(0.0, min(1.0, (by + bh / 2.0) / img_h))
                w = max(0.0, min(1.0, bw / img_w))
                h = max(0.0, min(1.0, bh / img_h))
                obj_dict.setdefault(stem, []).append(f"{yolo_cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return segm_dict, obj_dict


def parse_obj_zip_to_dict(zip_path):
    """
    MakeSense Object Detection .zip dosyasının içindeki .txt dosyalarını
    RAM üzerinde okur. Boş olan .txt dosyalarını kesinlikle almaz.
    Unicode NFD/NFC normalizasyonu uygular.
    """
    obj_dict = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/") or not member.lower().endswith(".txt"):
                    continue
                stem = normalize_stem(Path(member).stem)
                if stem.lower() in CLASS_DEF_NAMES:
                    continue

                raw_text = zf.read(member).decode("utf-8", errors="ignore")
                valid_lines = []
                for line in raw_text.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        valid_lines.append(" ".join(parts))

                if valid_lines:
                    obj_dict.setdefault(stem, []).extend(valid_lines)
    except Exception as e:
        print(f" [!] ZIP okuma hatası ({zip_path.name}): {e}")
    return obj_dict


def scan_source_folders(root_source_dir, segm_classes):
    """
    Hem alt klasörlü (Grup1, grup2, ...) hem de düz TEK klasörlü
    (görseller, .txt, .json ve .zip aynı klasörde) yapıları kusursuz tarar.
    macOS / Windows / Linux Türkçe karakter ve Unicode farklarını giderir.
    Etiketsiz görselleri veri setine kesinlikle dahil etmez.
    """
    root_path = Path(root_source_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: '{root_path}'")

    candidate_dirs = [root_path] + [d for d in root_path.rglob("*") if d.is_dir()]
    active_dirs = []
    for d in candidate_dirs:
        has_images = any(f.suffix.lower() in SUPPORTED_IMG_EXTS for f in d.iterdir() if f.is_file())
        if has_images:
            active_dirs.append(d)

    print(f"\n[Tarama]: '{root_path}' altında görsel içeren {len(active_dirs)} klasör tespit edildi.")

    dataset_items = []
    seen_stems = set()
    total_scanned_images = 0
    skipped_unlabeled_images = 0

    # Kök klasördeki olası genel etiket dosyaları (alt klasörlerin faydalanabilmesi için)
    root_global_zips = list(root_path.glob("*.zip"))
    root_global_jsons = list(root_path.glob("*.json"))

    for folder in active_dirs:
        folder_name = folder.name
        images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXTS]

        # 1. Klasörün kendi doğrudan etiket dosyaları
        zip_files = list(folder.glob("*.zip"))
        json_files = list(folder.glob("*.json"))
        loose_txt_files = [
            f for f in folder.glob("*.txt")
            if normalize_stem(f.stem).lower() not in CLASS_DEF_NAMES
        ]

        # 2. Eğer klasörün içinde görsel içermeyen alt klasörler varsa (Örn: 'labels_my-project-name...', 'labels', 'annotations')
        for sub_dir in folder.rglob("*"):
            if sub_dir.is_dir() and sub_dir not in active_dirs:
                zip_files.extend(list(sub_dir.glob("*.zip")))
                json_files.extend(list(sub_dir.glob("*.json")))
                loose_txt_files.extend([
                    f for f in sub_dir.glob("*.txt")
                    if normalize_stem(f.stem).lower() not in CLASS_DEF_NAMES
                ])

        # 3. Eğer alt klasörde hiç zip/json yoksa kök dizindeki genel zip/json'ları ekle
        if folder != root_path:
            for gz in root_global_zips:
                if gz not in zip_files:
                    zip_files.append(gz)
            for gj in root_global_jsons:
                if gj not in json_files:
                    json_files.append(gj)

        local_obj_labels = {}
        local_segm_labels = {}

        # 1. ZIP dosyalarından Object Detection etiketlerini al (NFC normalized)
        for zf in zip_files:
            parsed_obj = parse_obj_zip_to_dict(zf)
            for k, v in parsed_obj.items():
                local_obj_labels.setdefault(k, []).extend(v)

        # 2. JSON dosyalarından Segmentation ve Detection etiketlerini al (NFC normalized)
        for jf in json_files:
            parsed_segm, parsed_obj = parse_coco_json_to_dict(jf, segm_classes)
            for k, v in parsed_segm.items():
                local_segm_labels.setdefault(k, []).extend(v)
            for k, v in parsed_obj.items():
                local_obj_labels.setdefault(k, []).extend(v)

        # 3. Açık .txt dosyalarını sütun sayısına göre (5 = Obj, >=7 = Segm) ayır
        for tf in loose_txt_files:
            stem = normalize_stem(tf.stem)
            try:
                with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
            except Exception as e:
                print(f" [!] .txt okunamadı ({tf.name}): {e}")
                continue

            obj_l, segm_l = [], []
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    obj_l.append(" ".join(parts))
                elif len(parts) >= 7:
                    segm_l.append(" ".join(parts))

            if obj_l:
                local_obj_labels.setdefault(stem, []).extend(obj_l)
            if segm_l:
                local_segm_labels.setdefault(stem, []).extend(segm_l)

        # 4. Sadece etiketi olan görselleri veri setine kabul et
        folder_kept = 0
        folder_skipped = 0
        for img_path in sorted(images):
            total_scanned_images += 1
            orig_stem = normalize_stem(img_path.stem)
            obj_lines = local_obj_labels.get(orig_stem, [])
            segm_lines = local_segm_labels.get(orig_stem, [])

            # KRİTİK KONTROL: Görselin hiçbir etiketi yoksa veri setine ALMA!
            if not obj_lines and not segm_lines:
                skipped_unlabeled_images += 1
                folder_skipped += 1
                continue

            unique_stem = orig_stem
            if unique_stem in seen_stems:
                unique_stem = f"{folder_name}_{orig_stem}"
            counter = 1
            while unique_stem in seen_stems:
                unique_stem = f"{folder_name}_{orig_stem}_{counter}"
                counter += 1
            seen_stems.add(unique_stem)

            dataset_items.append({
                "img_path": img_path,
                "stem": unique_stem,
                "ext": img_path.suffix,
                "group": folder_name,
                "obj_lines": obj_lines,
                "segm_lines": segm_lines
            })
            folder_kept += 1

        print(
            f" -> Klasör: '{folder_name}' | Taranan: {len(images)} Görsel | "
            f"Eşleşen (Etiketli): {folder_kept} | Atlanan (Etiketsiz): {folder_skipped}"
        )

    return dataset_items, total_scanned_images, skipped_unlabeled_images


def get_split_ratios():
    """Kullanıcıdan Train, Val, Test oranlarını alır (Varsayılan: %70 Train, %20 Val, %10 Test)."""
    print("\n--- VERİ SETİ BÖLME ORANLARI (Varsayılan: Train %70, Val %20, Test %10) ---")
    custom = input("[?] Oranları değiştirmek ister misiniz? (e/h) [Varsayılan: h]: ").strip().lower()
    if custom != "e":
        return 0.70, 0.20, 0.10

    while True:
        try:
            tr = float(input(" -> Train yüzdesi (Örn: 70): ").strip() or 70)
            vl = float(input(" -> Val yüzdesi   (Örn: 20): ").strip() or 20)
            ts = float(input(" -> Test yüzdesi  (Örn: 10): ").strip() or 10)
            if abs((tr + vl + ts) - 100.0) > 1e-4:
                print("Hata: Oranların toplamı 100 olmalıdır! Lütfen tekrar girin.")
                continue
            return tr / 100.0, vl / 100.0, ts / 100.0
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")


def split_items_stratified(items, train_r, val_r):
    """
    Hem Object Detection hem de Segmentation etiketlerinin train/val/test
    klasörlerine tam oranda (%70-%20-%10) dağılmasını sağlar.
    """
    random.seed(42)
    both_labeled = [it for it in items if it["obj_lines"] and it["segm_lines"]]
    only_obj = [it for it in items if it["obj_lines"] and not it["segm_lines"]]
    only_segm = [it for it in items if not it["obj_lines"] and it["segm_lines"]]

    splits = {"train": [], "val": [], "test": []}

    for group in [both_labeled, only_obj, only_segm]:
        random.shuffle(group)
        n = len(group)
        t_end = int(round(n * train_r))
        v_end = t_end + int(round(n * val_r))
        splits["train"].extend(group[:t_end])
        splits["val"].extend(group[t_end:v_end])
        splits["test"].extend(group[v_end:])

    return splits


def build_complete_dataset(items, output_dir, train_r, val_r, aug_config, obj_classes=None, segm_classes=None):
    """
    Görsellerin orijinal boyutlarına dokunmadan klasörleri oluşturur.
    Asla boş .txt dosyası oluşturmaz.
    """
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)

    splits = split_items_stratified(items, train_r, val_r)

    for split_name, split_items in splits.items():
        img_dir = out_path / "images" / split_name
        obj_dir = out_path / "labels_obj" / split_name
        segm_dir = out_path / "labels_segm" / split_name

        img_dir.mkdir(parents=True, exist_ok=True)
        obj_dir.mkdir(parents=True, exist_ok=True)
        segm_dir.mkdir(parents=True, exist_ok=True)

        for item in split_items:
            stem = item["stem"]
            ext = item["ext"]

            # 1. Orijinal görseli olduğu gibi (çözünürlüğüne dokunmadan) kopyala
            dst_img = img_dir / f"{stem}{ext}"
            shutil.copy2(item["img_path"], dst_img)

            # 2. Sadece Object Detection etiketi varsa .txt oluştur (Boş dosya oluşturulmaz)
            if item["obj_lines"]:
                dst_obj = obj_dir / f"{stem}.txt"
                with open(dst_obj, "w", encoding="utf-8") as f:
                    f.write("\n".join(item["obj_lines"]) + "\n")

            # 3. Sadece Segmentation etiketi varsa .txt oluştur (Boş dosya oluşturulmaz)
            if item["segm_lines"]:
                dst_segm = segm_dir / f"{stem}.txt"
                with open(dst_segm, "w", encoding="utf-8") as f:
                    f.write("\n".join(item["segm_lines"]) + "\n")

            # 4. Sadece TRAIN seti için Veri Çoğaltma (Augmentation) uygula
            if split_name == "train" and aug_config.get("enabled", False):
                aug_results = apply_augmentations_to_item(
                    item["img_path"], item["obj_lines"], item["segm_lines"], aug_config
                )
                for suffix, aug_img, aug_obj, aug_segm in aug_results:
                    if not aug_obj and not aug_segm:
                        continue

                    aug_stem = f"{stem}_{suffix}"
                    aug_img_path = img_dir / f"{aug_stem}{ext}"

                    is_success, im_buf = cv2.imencode(ext, aug_img)
                    if is_success:
                        im_buf.tofile(str(aug_img_path))

                    if aug_obj:
                        with open(obj_dir / f"{aug_stem}.txt", "w", encoding="utf-8") as f:
                            f.write("\n".join(aug_obj) + "\n")

                    if aug_segm:
                        with open(segm_dir / f"{aug_stem}.txt", "w", encoding="utf-8") as f:
                            f.write("\n".join(aug_segm) + "\n")

    # Sınıf tanımlarını dataset klasörüne de kaydet (2_mod_degistir.py doğrudan okuyabilsin)
    if obj_classes:
        with open(out_path / "class.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(obj_classes) + "\n")
    if segm_classes:
        with open(out_path / "class_segm.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(segm_classes) + "\n")


def analyze_and_print_statistics(dataset_dir, obj_classes, segm_classes, scanned_count, skipped_count):
    """
    Görsel sayılarını, labels_obj ve labels_segm dosya sayılarını, toplam etiket
    adetlerini ve sınıf bazlı dağılımları detaylı olarak raporlar.
    """
    ds_path = Path(dataset_dir)
    splits = ["train", "val", "test"]

    img_counts = {}
    for s in splits:
        p = ds_path / "images" / s
        img_counts[s] = len([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXTS]) if p.exists() else 0

    def analyze_label_folder(folder_key):
        total_txt = 0
        total_annotations = 0
        class_counter = Counter()
        split_class_counter = {s: Counter() for s in splits}
        split_txt_counts = {}

        for s in splits:
            lbl_dir = ds_path / folder_key / s
            if not lbl_dir.exists():
                split_txt_counts[s] = 0
                continue
            txt_files = list(lbl_dir.glob("*.txt"))
            split_txt_counts[s] = len(txt_files)
            total_txt += len(txt_files)

            for tf in txt_files:
                with open(tf, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                for line in lines:
                    parts = line.split()
                    if parts:
                        cid = int(parts[0])
                        total_annotations += 1
                        class_counter[cid] += 1
                        split_class_counter[s][cid] += 1

        return {
            "total_txt": total_txt,
            "total_annotations": total_annotations,
            "class_counter": class_counter,
            "split_class_counter": split_class_counter,
            "split_txt_counts": split_txt_counts
        }

    obj_stats = analyze_label_folder("labels_obj")
    segm_stats = analyze_label_folder("labels_segm")

    line_sep = "=" * 72
    print("\n" + line_sep)
    print("                DETAYLI VERİ SETİ İSTATİSTİK RAPORU")
    print(line_sep)

    # 1. Görsel Sayıları
    total_imgs = sum(img_counts.values())
    print("\n[1] GÖRSEL SAYILARI (Orijinal Çözünürlük Korundu & Etiketsizler Elendi):")
    print(f"    • Kaynakta Taranan Ham Görsel     : {scanned_count}")
    print(f"    • Etiketsiz Olduğu İçin Atlanan   : {skipped_count} (Veri setine alınmadı)")
    print(f"    • Veri Setindeki Toplam Görsel    : {total_imgs} (Çoğaltılmış görseller dahil)")
    print(f"      - Train Klasörü                 : {img_counts['train']} görsel")
    print(f"      - Val Klasörü                   : {img_counts['val']} görsel")
    print(f"      - Test Klasörü                  : {img_counts['test']} görsel")

    # 2. Object Detection İstatistikleri
    print("\n[2] OBJECT DETECTION (labels_obj) İSTATİSTİKLERİ:")
    print(
        f"    • Obj .txt Dosya Sayısı           : {obj_stats['total_txt']} "
        f"(Train: {obj_stats['split_txt_counts']['train']}, "
        f"Val: {obj_stats['split_txt_counts']['val']}, "
        f"Test: {obj_stats['split_txt_counts']['test']})"
    )
    print(f"    • Obj Toplam Etiket (Kutu) Sayısı : {obj_stats['total_annotations']}")
    print("    • Obj Etiket Türleri ve Dağılımı:")
    for idx, cname in enumerate(obj_classes):
        tr_c = obj_stats["split_class_counter"]["train"][idx]
        vl_c = obj_stats["split_class_counter"]["val"][idx]
        ts_c = obj_stats["split_class_counter"]["test"][idx]
        tot_c = obj_stats["class_counter"][idx]
        print(f"      - Sınıf {idx} ({cname:<15}): Toplam = {tot_c:<5} | (Train: {tr_c}, Val: {vl_c}, Test: {ts_c})")

    # 3. Segmentation İstatistikleri
    print("\n[3] INSTANCE SEGMENTATION (labels_segm) İSTATİSTİKLERİ:")
    print(
        f"    • Segm .txt Dosya Sayısı          : {segm_stats['total_txt']} "
        f"(Train: {segm_stats['split_txt_counts']['train']}, "
        f"Val: {segm_stats['split_txt_counts']['val']}, "
        f"Test: {segm_stats['split_txt_counts']['test']})"
    )
    print(f"    • Segm Toplam Etiket (Poligon)    : {segm_stats['total_annotations']}")
    print("    • Segm Etiket Türleri ve Dağılımı:")
    for idx, cname in enumerate(segm_classes):
        tr_c = segm_stats["split_class_counter"]["train"][idx]
        vl_c = segm_stats["split_class_counter"]["val"][idx]
        ts_c = segm_stats["split_class_counter"]["test"][idx]
        tot_c = segm_stats["class_counter"][idx]
        print(f"      - Sınıf {idx} ({cname:<15}): Toplam = {tot_c:<5} | (Train: {tr_c}, Val: {vl_c}, Test: {ts_c})")

    print(line_sep)
    print("Tamamlandı! Eğitim öncesi aktif modu seçmek için '2_mod_degistir.py' çalıştırın.\n")


if __name__ == "__main__":
    KAYNAK_KLASOR = "./zaman_damgali_frameler"
    HEDEF_DATASET = "./dataset"
    OBJ_SINIF_DOSYASI = "./class.txt"
    SEGM_SINIF_DOSYASI = "./class_segm.txt"

    obj_classes = load_class_list(OBJ_SINIF_DOSYASI, ["insan"], search_dirs=[KAYNAK_KLASOR])
    segm_classes = load_class_list(SEGM_SINIF_DOSYASI, ["diseased_leaf", "healthy_leaf"], search_dirs=[KAYNAK_KLASOR])

    # 1. Kaynak klasörü tara ve etiketsiz görselleri anında ele
    items, scanned_total, skipped_total = scan_source_folders(KAYNAK_KLASOR, segm_classes)

    if not items:
        print("Hata: Hiçbir etiketli görsel bulunamadı! Lütfen zip ve json dosyalarını kontrol edin.")
    else:
        # 2. Bölme oranlarını ve Veri Çoğaltma tercihlerini al
        train_ratio, val_ratio, test_ratio = get_split_ratios()
        aug_cfg = get_augmentation_config()

        # 3. Veri setini oluştur
        build_complete_dataset(items, HEDEF_DATASET, train_ratio, val_ratio, aug_cfg, obj_classes=obj_classes, segm_classes=segm_classes)

        # 4. Detaylı istatistikleri ekrana yazdır
        analyze_and_print_statistics(HEDEF_DATASET, obj_classes, segm_classes, scanned_total, skipped_total)