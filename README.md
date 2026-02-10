# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București – FIIR

**Student:** Samata George

**Data:** 22.01.2026

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care s-a analizat, curățat și augmentat setul de date pentru clasificarea defectelor de sudură. Scopul a fost transformarea datelor brute (dezorganizate) într-un set structurat, echilibrat și suficient de mare (4000+ imagini) pentru a antrena o rețea convoluțională (CNN).

---

## 1. Structura Repository-ului Github (versiunea Etapei 3)

```
Sistem AI pentru Clasificarea Defectelor de Sudura/
├── README_Etapa3.md       # Acest fișier
├── data/
│   ├── raw/               # Datele brute organizate (cele 5 clase curate)
│   ├── train/             # Setul FINAL (Originale + Generate) folosit la antrenare
│   └── generated/         # Copie a datelor sintetice (pentru evidențierea contribuției)
├── src/
│   ├── data_acquisition/  
│   │   ├── fix_dataset.py    # Script curățare foldere Roboflow
│   │   └── generate_data.py  # Script augmentare (zgomot + luminozitate)
│   └── neural_network/    
│       └── train.py          # Scriptul care încarcă datele
├── docs/
│   └── confusion_matrix.png
└── requirements.txt

```

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Dataset public Roboflow ("Welding Defect") combinat cu date sintetice.
* **Modul de achiziție:** * [X] Fișier extern (Roboflow) - baza de imagini reale.
* [X] Generare programatică - augmentare avansată pentru triplarea dataset-ului.


* **Perioada / condițiile colectării:** Ianuarie 2026.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** **4116 imagini** (1372 originale + 2744 generate).
* **Număr de caracteristici (features):** 3 (Înălțime 224 x Lățime 224 x 3 Canale RGB).
* **Tipuri de date:** [X] Imagini (`.jpg`, `.png`).
* **Clase (Target):** 5 clase de defecte.

### 2.3 Descrierea caracteristicilor (Input/Output)

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
| --- | --- | --- | --- | --- |
| **Input Image** | Matrice 3D | Pixeli | Imaginea sudurii redimensionată la 224x224 | 0–255 (int) |
| **Label (Clasa)** | Categorial | - | Tipul defectului: `bad_weld`, `crack`, `good_weld`, `porosity`, `spatter` | {0, 1, 2, 3, 4} |

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Probleme identificate inițial

În faza de analiză a datelor brute descărcate de pe Roboflow, s-au identificat următoarele probleme critice:

1. **Structură Haotică:** Dataset-ul conținea peste **90 de foldere** redundante generate automat (ex: `bad_weld_multiple_spatter`, `good_weld_spatter`), în loc de cele 5 clase principale.
2. **Dezechilibru de clasă:** Clasa `spatter` avea sute de imagini, în timp ce clasa `crack` avea foarte puține exemplare inițiale.
3. **Variații de mediu:** Imaginile proveneau din surse diferite, având iluminare și rezoluții inconsistente.

### 3.2 Statistici descriptive (Post-Procesare)

În urma curățării, distribuția a fost stabilizată, iar toate imaginile au fost aduse la standardul CNN.

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea datelor (Script `fix_dataset.py`)

Am implementat un script custom (`src/data_acquisition/fix_dataset.py`) care a rezolvat problema folderelor multiple:

* **Algoritm:** A scanat recursiv directoarele `raw`, `train`, `test` de la Roboflow.
* **Acțiune:** A identificat cuvinte cheie în numele folderelor și a mutat automat imaginile în cele 5 clase canonice: `bad_weld`, `crack`, `good_weld`, `porosity`, `spatter`.
* **Rezultat:** 1372 imagini salvate corect, 94 foldere inutile șterse.

### 4.2 Transformarea și Augmentarea (Script `generate_data.py`)

Pentru a îndeplini cerința de originalitate și a îmbunătăți robustețea modelului, am aplicat:

1. **Zgomot Gaussian:** Simulează senzori industriali de slabă calitate.
2. **Variații de Luminozitate (HSV):** Simulează condiții de sudură în întuneric sau lumină puternică.
3. **Redimensionare:** Toate imaginile au fost aduse la **224x224 px**.

**Rezultat Augmentare:** Dataset-ul a crescut de la 1372 la **4116 imagini** (66.67% contribuție proprie).

### 4.3 Structurarea pentru Antrenare

Datele finale se află în `data/train/`. Împărțirea în seturi de antrenare și validare se face dinamic în cod (`train.py`) folosind Keras:

* **80% Antrenare**
* **20% Validare** (Stratificat)

### 4.4 Transformări la încărcare (Normalization)

În pipeline-ul de antrenare, se aplică un strat de normalizare:

```python
normalization_layer = tf.keras.layers.Rescaling(1./255)

```

Acesta transformă valorile pixelilor din `[0, 255]` în `[0, 1]` pentru convergență rapidă.

---

## 5. Fișiere Generate în Această Etapă

* `data/raw/` – Cele 5 foldere curățate.
* `data/train/` – Dataset-ul complet (4116 imagini).
* `src/data_acquisition/fix_dataset.py` – Scriptul de curățare a structurii.
* `src/data_acquisition/generate_data.py` – Scriptul de generare a datelor sintetice.

---

## 6. Stare Etapă

* [X] Structură repository configurată
* [X] Dataset analizat și curățat de erori structurale
* [X] Date preprocesate (Resize 224x224)
* [X] Augmentare realizată (66% date originale generate)
* [X] Documentație actualizată

---