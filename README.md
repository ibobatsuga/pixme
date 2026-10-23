# PixMe 🏃📸

Platform pencarian dan galeri dokumentasi foto event berbasis **AI Face Recognition & Vector Matching** (alternatif modern & efisien untuk FotoYu), dirancang khusus untuk berjalan dengan **$0 / Free-Tier**.

---

## ⚡ Fitur Utama

1. **AI Face Recognition Cepat (CPU-Only):**
   - Deteksi wajah multi-sudut menggunakan model ONNX **YuNet**.
   - Ekstraksi vektor embedding 128-dimensi menggunakan model ONNX **SFace**.
   - Tidak memerlukan GPU mahal — berjalan super cepat di laptop atau VPS gratis.
2. **Pencarian Foto dengan Selfie:**
   - Peserta event cukup mengambil selfie melalui kamera ponsel atau mengunggah foto wajah.
   - Sistem mencocokkan kemiripan vektor (*cosine similarity*) dan menampilkan seluruh foto dirinya secara instan.
3. **Portal Fotografer (Bulk Ingest):**
   - Fotografer dapat mengunggah puluhan hingga ratusan foto event sekaligus.
   - Thumbnail web otomatis dibuat untuk menghemat bandwidth.
   - Perceptual Hash (pHash) untuk mencegah duplikasi foto.
4. **100% Bebas Biaya (Tanpa Paywall di Awal):**
   - Pengguna dapat langsung mengunduh foto kualitas asli (*Full HD*) secara gratis.

---

## 🚀 Cara Menjalankan Aplikasi

Pastikan Anda berada di direktori proyek:
```bash
cd /Users/ibobatsuga/.gemini/antigravity/scratch/eventphoto-ai
```

Jalankan server aplikasi:
```bash
./run.sh
```
Atau secara manual:
```bash
.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Buka peramban (browser) di:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📁 Struktur Direktori

```
pixme/
├── backend/
│   └── app/
│       ├── main.py               # REST API & static file serving
│       └── services/
│           ├── face_engine.py    # Modul YuNet + SFace face recognition
│           ├── storage.py        # Local & Cloudflare R2 storage handler
│           └── database.py       # SQLite & pgvector data layer
├── database/
│   └── schema.sql                # Skema Supabase pgvector untuk deploy cloud
├── frontend/
│   └── index.html                # Antarmuka web modern (Tailwind + Vanilla JS)
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── uploads/                      # Direktori penyimpanan foto & thumbnail lokal
├── requirements.txt
└── run.sh
```

---

## ☁️ Langkah Deploy Cloud ($0 Free Tier)

1. **Database:** Buat proyek gratis di [Supabase.com](https://supabase.com), buka menu SQL Editor, dan jalankan perintah dari file [`database/schema.sql`](database/schema.sql).
2. **Storage:** Buat bucket gratis di [Cloudflare R2](https://dash.cloudflare.com) untuk penyimpanan foto beresolusi tinggi tanpa biaya egress.
3. **Hosting:** Deploy backend Python ini ke [Render.com](https://render.com) atau [Koyeb.com](https://koyeb.com) (keduanya menyediakan Free Tier).
