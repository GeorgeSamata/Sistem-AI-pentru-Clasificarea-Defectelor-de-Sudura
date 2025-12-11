# Sistem AI pentru Clasificarea Defectelor de Sudură

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Șamata George Cristian
**Data:** 03.12.2025

---

## 1. Descrierea Problemei

Proiectul vizează dezvoltarea unui sistem avansat de vizualizare computerizată (Computer Vision) pentru controlul nedistructiv al calității (NDT). Obiectivul este detectarea automată și localizarea defectelor în cordoanele de sudură industriale folosind arhitectura **YOLOv8**.

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Dataset public adaptat din "Welding Defect Object Detection" (Kaggle), conținând imagini radiografice și macro ale sudurilor.
* **Modul de achiziție:** [ ] Senzori reali / [ ] Simulare / [X] Fișier extern (Kaggle) / [X] Generare programatică (Augmentare date)
* **Perioada / condițiile colectării:** Noiembrie 2024 - Ianuarie 2025 (Selectarea și etichetarea manuală a datelor relevante pentru sudură).

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 2,500 imagini (estimat după augmentare)
* **Număr de caracteristici (features):** 3 (Înălțime x Lățime x Canale RGB)
* **Tipuri de date:** [X] Numerice (Coordonate bbox) / [ ] Categoriale / [ ] Temporale / [X] Imagini
* **Format fișiere:** [ ] CSV / [X] TXT (Adnotări YOLO) / [ ] JSON / [X] PNG/JPG / [ ] Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| **INPUT 1:** Matrice Pixeli | Numeric (Tensor) | Intensitate (0-255) | Informația vizuală brută a imaginii pe 3 canale (RGB). | [0, 255] |
| **INPUT 2:** Lățime Imagine | Numeric | Pixeli (px) | Rezoluția orizontală la care este redimensionată imaginea pentru rețea. | Fix: 640 |
| **INPUT 3:** Înălțime Imagine | Numeric | Pixeli (px) | Rezoluția verticală la care este redimensionată imaginea. | Fix: 640 |
| **OUTPUT 1:** BBox Center (x,y) | Numeric | Coordonate Relative | Poziția centrului geometric al defectului detectat. | [0.0, 1.0] |
| **OUTPUT 2:** BBox Size (w,h) | Numeric | Dimensiuni Relative | Lățimea și înălțimea dreptunghiului care încadrează defectul. | [0.0, 1.0] |
| **OUTPUT 3:** Confidence Score | Numeric | Probabilitate | Gradul de certitudine al modelului pentru detecția curentă. | [0.0, 1.0] |
| **OUTPUT 4:** Class ID | Categorial | ID Etichetă | Tipul defectului identificat (ex: Porozitate, Fisură). | {0, 1, 2, 3, 4} |

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distribuția claselor:** Analiza numărului de instanțe pentru fiecare tip de defect (ex: "Bad Welding" vs "Good Welding") pentru a verifica balansul datelor.
* **Distribuția dimensiunilor BBox:** Histograme ale dimensiunilor defectelor (mici vs. mari) pentru a ajusta ancorele modelului (dacă e cazul).
* **Harta termică a pozițiilor:** Vizualizarea zonelor din imagine unde apar cel mai frecvent defectele.

### 3.2 Analiza calității datelor

* **Verificarea integrității imaginilor:** Identificarea fișierelor corupte care nu pot fi deschise de OpenCV.
* **Verificarea etichetelor:** Identificarea fișierelor imagine care nu au un fișier `.txt` asociat sau au coordonate invalide (în afara [0,1]).

### 3.3 Probleme identificate

* [Identificat] Dezechilibru de clasă: Mai multe exemple de "Good Welding" decât "Crack".
* [Identificat] Variații mari de iluminare în imaginile preluate din surse diferite.

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicate:** Ștergerea imaginilor identice (hash-checking).
* **Tratarea etichetelor lipsă:** Eliminarea imaginilor din setul de antrenament care nu conțin adnotări valide.

### 4.2 Transformarea caracteristicilor

* **Redimensionare:** Toate imaginile sunt redimensionate la 640x640 px (standard YOLOv8) cu padding (letterbox) pentru a păstra raportul de aspect.
* **Normalizare:** Valorile pixelilor (0-255) sunt împărțite la 255 pentru a obține intervalul [0, 1].
* **Augmentare:** Aplicarea de rotații, flip-uri și ajustări de luminozitate "on-the-fly" în timpul antrenării (Mosaic augmentation specific YOLO).

### 4.3 Structurarea seturilor de date

**Împărțire realizată:**
* 70% – train (pentru învățarea parametrilor)
* 20% – validation (pentru tuning-ul hiperparametrilor în timpul epocilor)
* 10% – test (pentru evaluarea finală nepolarizată)

**Principii respectate:**
* Imaginile au fost amestecate (shuffled) înainte de împărțire.
* Nu există suprapuneri între seturile de Train și Test (Data Leakage prevention).

### 4.4 Salvarea rezultatelor preprocesării

* Configurația dataset-ului este salvată în `data.yaml`.
* Structura de foldere respectă standardul Ultralytics: `images/train`, `labels/train`, etc.

---

## 5. Fișiere Generate în Această Etapă

* `data/raw/` – imaginile originale descărcate.
* `data/processed/` – (virtual) imaginile sunt procesate în timp real de dataloader-ul YOLO, dar structura logică este definită în `data.yaml`.
* `src/train_yolo.py` – scriptul de antrenare.
* `data/README.md` – descrierea detaliată a sursei și structurii.

---

## 6. Stare Etapă (de completat de student)

- [X] Structură repository configurată
- [X] Dataset analizat (EDA realizată) - *În curs*
- [X] Configurare `data.yaml` și structură foldere YOLO
- [X] Antrenare model (Epochs 1-50)
- [X] Documentație actualizată în README

---
```