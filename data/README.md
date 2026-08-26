# Direktori Data

Data mentah, data sementara, dan data hasil pemrosesan tidak disimpan di Git.

Proyek hanya menggunakan Base dataset dari Bank Account Fraud Dataset Suite
selama pengerjaan versi utama. Petunjuk pengunduhan, nama file aktual, checksum,
dan schema akan ditambahkan pada Fase 1.

Sumber dataset:

https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

Struktur lokal yang direncanakan:

- data/raw: data asli yang tidak diubah.
- data/interim: hasil transformasi sementara.
- data/processed: data siap digunakan oleh pipeline.
