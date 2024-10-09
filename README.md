# CHATBOT SHAVIRA (GANESHA VIRTUAL ASSISTANT)

## Direktori File SHAVIRA
- `SHAVIRA UNDIKSHA/`
  - `agents/`: Direktori untuk menyimpan fungsi tiap agent
    - `children/`: Menampung agen-agen dibawah question identifier
      - `sso/`: Menampung agen untuk reset akun SSO Undiksha
        - `sso_agent`: Menangani reset password akun Undiksha
      - `academic_agent.py`: Menangani pertanyaan terkait informasi akademis Undiksha
      - `account_agent.py`: Menangani konteks reset akun Undiksha
      - `general_agent.py`: Menangani pertanyaan umum terkait SHAVIRA dan Undiksha
      - `news_agent.py`: Menangani pertanyaan terkait berita Undiksha
      - `out_of_context_agent.py`: Menangani pertanyaan diluar konteks Undiksha dan layanan helpdesk
      - `student_agent.py`: Menangani pertanyaan terkait kegiatan kemahasiswaan di Undiksha
    - `__init__.py`: Inisiasi fungsi agar bisa diakses dari luar
    - `question_identifier_agent.py`: Agen pertama untuk menentukan konteks pertanyaan user
    - `writter_agent.py`: Agen terakhir yang memberikan output kepada user
  - `assets/`: Berisi asset seperti untuk SHAVIRA
    - `datasets/`: berisi dataset untuk agen-agen SHAVIRA
    - `images/`: memuat gambar flowmap SHAVIRA
  - `config/`: menyimpan file konfigurasi seperti prompt yang akan digunakan
  - `graph/`: menampilkan visualisasi graf untuk satu pertanyaan
  - `models/`: menampung model agen state yang akan digunakan pada tiap node (agen)
  - `utils/`: menampung fungsi eksternal untuk keperluan agen
  - `.env.example`: menyimpan file penting seperti token ChatGPT dan endpoint Ollama server
  - `.gitignore`: Mengabaikan file krusial untuk di upload di github
  - `app.py`: Membuat interface streamlit
  - `main.py`: Kode utama untu menjalankan Shavira
  - `READMW.MD`: Dokumentasi SHAVIRA
  - `requiements.txt`: Menampung modul python yang harus diinstal ketika menjalankan shavira
  - `tes.py`: File untuk testing kode-kode baru
## Ringkasan

![Roundmap Langgrap Shavira](/assets/images/SHAVIRA%20ROUNDMAP.jpg)
Proyek ini mengembangkan chatbot berbasis Retrieval Augmented Generation (RAG) untuk Sistem Helpdesk di Universitas Pendidikan Ganesha (Undiksha). Chatbot ini dirancang untuk memberikan informasi yang akurat dan cepat, meningkatkan efisiensi dan pengalaman pengguna dalam menggunakan sistem helpdesk.

## Apa itu RAG?

![image](https://gradientflow.com/wp-content/uploads/2023/10/newsletter87-RAG-simple.png)
Retrieval-Augmented Generation (RAG) adalah teknik yang dirancang untuk meningkatkan kinerja Large Language Model (LLM) dengan mengakses informasi dari sumber eksternal. Dengan RAG, chatbot dapat memberikan jawaban yang lebih akurat dan relevan, serta mengurangi kemungkinan halusinasi terhadap suatu informasi.

## Alur Kerja RAG

#### 1. Retrieve (Kumpulkan):

- Kueri dari pengguna digunakan untuk mencari konteks relevan dari sumber pengetahuan eksternal.
- Kuery diubah menjadi vektor dan dicocokkan dengan vektor dalam database (sumber pengetahuan juga telah melewati fase ini), sehingga mendapatkan objek data relevan (k untuk objek paling relevan).

#### 2. Augment (Tambahkan):

- Konteks diambil dan digabungkan dengan kueri pengguna menggunakan template prompt.

#### 3. Generate (Hasilkan Respon):

- Prompt yang sudah dimodifikasi struktur datanya dimasukkan ke dalam LLM untuk menghasilkan respons akhir.

## Contoh Implementasi

![image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*h5SO9Hqu1YVYQVEIsWGZBg.png)
Pertanyaan Pengguna (Kueri) "Apa syarat untuk mendaftar sebagai mahasiswa baru di Undiksha?"

#### 1. Retrieve

Konteks relevan diambil dari database vektor.
Konteks: "Untuk mendaftar sebagai mahasiswa baru di Undiksha, calon mahasiswa harus memiliki ijazah SMA atau sederajat, melengkapi formulir pendaftaran, dan mengikuti ujian masuk."

#### 2. Augment

Gabungkan kuery pengguna dengan konteks yang diambil menggunakan template prompt.
Prompt: "Syarat-syarat pendaftaran mahasiswa baru di UndikshaSyarat-syarat pendaftaran mahasiswa baru di Undiksha adalah sebagai berikut: sebagai berikut: {context}"

#### 3. Generate

LLM memproses prompt tersebut untuk menghasilkan respons lengkap.
Respons Akhir: "Syarat-syarat pendaftaran mahasiswa baru di Undiksha adalah sebagai berikut: Untuk mendaftar sebagai mahasiswa baru di Undiksha, calon mahasiswa harus memiliki ijazah SMA atau sederajat, melengkapi formulir pendaftaran, dan mengikuti ujian masuk."

## Referensi

- [From Basics to Advanced: Exploring LangGraph](https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787)
- [Build a Reliable RAG Agent using LangGraph](https://medium.com/the-ai-forum/build-a-reliable-rag-agent-using-langgraph-2694d55995cd)

Developed By [DiarCode11](https://github.com/DiarCode11) & [odetv](https://github.com/odetv)
