import cv2
import numpy as np

# =====================================================================
# YARDIMCI KOORDİNAT DÖNÜŞTÜRÜCÜLER (OBJ VE SEGM İÇİN)
# =====================================================================

def _transform_obj_lines(obj_lines, flip_x=False, flip_y=False):
    """Object Detection (class x_center y_center w h) koordinatlarını günceller."""
    new_lines = []
    for line in obj_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = parts[0]
        xc, yc, w, h = map(float, parts[1:])
        if flip_x:
            xc = max(0.0, min(1.0, 1.0 - xc))
        if flip_y:
            yc = max(0.0, min(1.0, 1.0 - yc))
        new_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return new_lines


def _transform_segm_lines(segm_lines, flip_x=False, flip_y=False):
    """Segmentation (class x1 y1 x2 y2 ... xn yn) poligon koordinatlarını günceller."""
    new_lines = []
    for line in segm_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cid = parts[0]
        coords = list(map(float, parts[1:]))
        new_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i + 1]
            if flip_x:
                x = max(0.0, min(1.0, 1.0 - x))
            if flip_y:
                y = max(0.0, min(1.0, 1.0 - y))
            new_coords.append(f"{x:.6f} {y:.6f}")
        new_lines.append(f"{cid} " + " ".join(new_coords))
    return new_lines


# =====================================================================
# 5 FARKLI ALANDA VERİ ÇOĞALTMA FONKSİYONU (BOYUT DEĞİŞTİRMEZ)
# =====================================================================

def aug_1_horizontal_flip(image, obj_lines, segm_lines):
    """1. Alan (Geometrik - Yatay): Görseli yatay çevirir, obj ve segm X koordinatlarını günceller."""
    aug_img = cv2.flip(image, 1)
    aug_obj = _transform_obj_lines(obj_lines, flip_x=True, flip_y=False)
    aug_segm = _transform_segm_lines(segm_lines, flip_x=True, flip_y=False)
    return aug_img, aug_obj, aug_segm


def aug_2_vertical_flip(image, obj_lines, segm_lines):
    """2. Alan (Geometrik - Dikey): Görseli dikey çevirir, obj ve segm Y koordinatlarını günceller."""
    aug_img = cv2.flip(image, 0)
    aug_obj = _transform_obj_lines(obj_lines, flip_x=False, flip_y=True)
    aug_segm = _transform_segm_lines(segm_lines, flip_x=False, flip_y=True)
    return aug_img, aug_obj, aug_segm


def aug_3_brightness_contrast(image, obj_lines, segm_lines, alpha=1.25, beta=20.0):
    """3. Alan (Fotometrik - Işık/Kontrast): Parlaklık (beta) ve kontrastı (alpha) ayarlar."""
    aug_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return aug_img, list(obj_lines), list(segm_lines)


def aug_4_hsv_color_jitter(image, obj_lines, segm_lines, h_shift=12.0, s_scale=1.25, v_scale=1.15):
    """4. Alan (Renk Uzayı - HSV): Yaprak ve nesne tonlarını/doygunluğunu değiştirir."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return aug_img, list(obj_lines), list(segm_lines)


def aug_5_gaussian_noise_blur(image, obj_lines, segm_lines, noise_sigma=12.0, blur_ksize=3):
    """5. Alan (Sensör/Optik - Gürültü ve Bulanıklık): Kamera gürültüsü ve hafif bulanıklık ekler."""
    aug_img = image.astype(np.float32)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, image.shape).astype(np.float32)
        aug_img = np.clip(aug_img + noise, 0, 255)
    aug_img = aug_img.astype(np.uint8)

    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

    return aug_img, list(obj_lines), list(segm_lines)


# =====================================================================
# İNTERAKTİF KULLANICI AYARLARI VE UYGULAYICI
# =====================================================================

def get_augmentation_config():
    """Kullanıcıya veri çoğaltma yapmak isteyip istemediğini ve parametreleri sorar."""
    config = {
        "enabled": False,
        "hflip": True,
        "vflip": True,
        "bc_enabled": True, "alpha": 1.25, "beta": 20.0,
        "hsv_enabled": True, "h_shift": 12.0, "s_scale": 1.25, "v_scale": 1.15,
        "noise_enabled": True, "noise_sigma": 12.0, "blur_ksize": 3
    }

    ans = input("\n[?] Eğitim (Train) setine Veri Çoğaltma (Augmentation) uygulansın mı? (e/h) [Varsayılan: h]: ").strip().lower()
    if ans != "e":
        print(" -> Veri çoğaltma devre dışı bırakıldı.")
        return config

    config["enabled"] = True
    custom = input("[?] Veri çoğaltma ayarlarını özelleştirmek ister misiniz? (e: Özelleştir / h: Varsayılanlar) [Varsayılan: h]: ").strip().lower()
    if custom != "e":
        print(" -> Varsayılan 5'li veri çoğaltma ayarları kullanılacak.")
        return config

    print("\n--- 5 FARKLI VERİ ÇOĞALTMA ALANI AYARLARI (Varsayılan için Enter'a basın) ---")

    v = input("1. Yatay Çevirme (Horizontal Flip) aktif olsun mu? (e/h) [e]: ").strip().lower()
    config["hflip"] = (v != "h")

    v = input("2. Dikey Çevirme (Vertical Flip) aktif olsun mu? (e/h) [e]: ").strip().lower()
    config["vflip"] = (v != "h")

    v = input("3. Parlaklık ve Kontrast değişimi aktif olsun mu? (e/h) [e]: ").strip().lower()
    config["bc_enabled"] = (v != "h")
    if config["bc_enabled"]:
        a = input("   -> Kontrast çarpanı (alpha) [Varsayılan: 1.25]: ").strip()
        b = input("   -> Parlaklık eklemesi (beta) [Varsayılan: 20]: ").strip()
        if a:
            config["alpha"] = float(a)
        if b:
            config["beta"] = float(b)

    v = input("4. HSV Renk/Ton/Doygunluk değişimi aktif olsun mu? (e/h) [e]: ").strip().lower()
    config["hsv_enabled"] = (v != "h")
    if config["hsv_enabled"]:
        h = input("   -> Ton kaydırma (Hue Shift 0-180) [Varsayılan: 12]: ").strip()
        s = input("   -> Doygunluk çarpanı (Saturation Scale) [Varsayılan: 1.25]: ").strip()
        val = input("   -> Parlaklık çarpanı (Value Scale) [Varsayılan: 1.15]: ").strip()
        if h:
            config["h_shift"] = float(h)
        if s:
            config["s_scale"] = float(s)
        if val:
            config["v_scale"] = float(val)

    v = input("5. Sensör Gürültüsü ve Bulanıklık aktif olsun mu? (e/h) [e]: ").strip().lower()
    config["noise_enabled"] = (v != "h")
    if config["noise_enabled"]:
        ns = input("   -> Gürültü şiddeti (Sigma) [Varsayılan: 12.0]: ").strip()
        bk = input("   -> Bulanıklık Kernel boyutu (Tek sayı: 1, 3, 5) [Varsayılan: 3]: ").strip()
        if ns:
            config["noise_sigma"] = float(ns)
        if bk:
            config["blur_ksize"] = int(bk)

    return config


def apply_augmentations_to_item(img_path, obj_lines, segm_lines, config):
    """
    Seçilen bir görsele aktif olan çoğaltma fonksiyonlarını uygular.
    Görselin orijinal piksel boyutları (WxH) kesinlikle değiştirilmez.
    """
    if not config.get("enabled", False):
        return []

    img_array = np.fromfile(str(img_path), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        return []

    results = []

    if config.get("hflip", True):
        im, ob, sg = aug_1_horizontal_flip(image, obj_lines, segm_lines)
        results.append(("aug_hflip", im, ob, sg))

    if config.get("vflip", True):
        im, ob, sg = aug_2_vertical_flip(image, obj_lines, segm_lines)
        results.append(("aug_vflip", im, ob, sg))

    if config.get("bc_enabled", True):
        im, ob, sg = aug_3_brightness_contrast(
            image, obj_lines, segm_lines,
            alpha=config["alpha"], beta=config["beta"]
        )
        results.append(("aug_bc", im, ob, sg))

    if config.get("hsv_enabled", True):
        im, ob, sg = aug_4_hsv_color_jitter(
            image, obj_lines, segm_lines,
            h_shift=config["h_shift"], s_scale=config["s_scale"], v_scale=config["v_scale"]
        )
        results.append(("aug_hsv", im, ob, sg))

    if config.get("noise_enabled", True):
        im, ob, sg = aug_5_gaussian_noise_blur(
            image, obj_lines, segm_lines,
            noise_sigma=config["noise_sigma"], blur_ksize=config["blur_ksize"]
        )
        results.append(("aug_noise", im, ob, sg))

    return results