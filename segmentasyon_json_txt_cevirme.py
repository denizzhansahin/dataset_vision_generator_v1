import json
from pathlib import Path

def convert_coco_folder_to_yolo_seg(json_folder_path, output_labels_dir):
    json_folder = Path(json_folder_path)
    output_dir = Path(output_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Klasördeki tüm .json dosyalarını bul
    json_files = list(json_folder.glob("*.json"))
    
    if not json_files:
        print(f"Uyarı: '{json_folder}' içinde hiç .json dosyası bulunamadı!")
        return

    for json_file in json_files:
        print(f"İşleniyor: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Kategorileri 0'dan başlayan YOLO indekslerine eşle
        # categories: [{"id": 1, "name": "diseased_leaf"}, {"id": 2, "name": "healthy_leaf"}]
        categories = sorted(data.get('categories', []), key=lambda x: x['id'])
        cat_id_to_yolo_id = {cat['id']: idx for idx, cat in enumerate(categories)}
        
        print("Sınıf Eşleştirmesi (COCO ID -> YOLO ID -> İsim):")
        for cat in categories:
            print(f"  COCO {cat['id']} -> YOLO {cat_id_to_yolo_id[cat['id']]} ({cat['name']})")

        # 2. Görsel bilgilerini (id -> {width, height, file_name}) sözlüğe al
        images_info = {}
        for img in data.get('images', []):
            images_info[img['id']] = img
            # Her görsel için başlangıçta boş bir .txt oluştur (etiketsiz görseller için önemli)
            txt_filename = Path(img['file_name']).stem + ".txt"
            txt_path = output_dir / txt_filename
            if not txt_path.exists():
                txt_path.touch()

        # 3. Etiketleri (annotations) görsel bazında grupla
        annotations_by_image = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # 4. Koordinatları normalize et ve .txt dosyalarına yaz
        converted_count = 0
        for img_id, anns in annotations_by_image.items():
            if img_id not in images_info:
                continue
                
            img_data = images_info[img_id]
            img_w = img_data['width']
            img_h = img_data['height']
            
            txt_filename = Path(img_data['file_name']).stem + ".txt"
            txt_path = output_dir / txt_filename

            lines = []
            for ann in anns:
                # Kalabalık (iscrowd=1 / RLE) formatındaki etiketleri atla (senin dosyanda hepsi 0)
                if ann.get('iscrowd', 0) == 1:
                    continue
                    
                yolo_class_id = cat_id_to_yolo_id[ann['category_id']]
                segmentation = ann.get('segmentation', [])
                
                for poly in segmentation:
                    # Poligon en az 3 noktadan (6 koordinat) oluşmalı
                    if len(poly) < 6:
                        continue
                        
                    norm_coords = []
                    for i in range(0, len(poly), 2):
                        # X ve Y koordinatlarını [0, 1] aralığına normalize et ve sınırla
                        x = max(0.0, min(1.0, poly[i] / img_w))
                        y = max(0.0, min(1.0, poly[i + 1] / img_h))
                        norm_coords.append(f"{x:.6f} {y:.6f}")
                    
                    line = f"{yolo_class_id} " + " ".join(norm_coords)
                    lines.append(line)

            # Dosyaya yaz (Eğer klasörde birden fazla JSON aynı görsele atıfta bulunuyorsa 'a' ile ekler)
            with open(txt_path, 'w', encoding='utf-8') as out_file:
                out_file.write("\n".join(lines) + ("\n" if lines else ""))
            
            converted_count += 1

        print(f"Tamamlandı! Toplam {len(images_info)} görsel için .txt dosyası '{output_dir}' klasörüne kaydedildi.\n")

# --- KULLANIM ---
if __name__ == "__main__":
    # JSON dosyanın (veya dosyalarının) bulunduğu klasör yolu
    JSON_KLASORU = "./json_dosyalari" 
    
    # Çıktı .txt dosyalarının kaydedileceği klasör yolu
    CIKTI_KLASORU = "./labels"        
    
    convert_coco_folder_to_yolo_seg(JSON_KLASORU, CIKTI_KLASORU)