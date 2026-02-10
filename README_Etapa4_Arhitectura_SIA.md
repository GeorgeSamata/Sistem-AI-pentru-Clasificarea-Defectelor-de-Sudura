# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București – FIIR

**Student:** Șamata George Cristian
**Data:** 22.01.2026

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN**.
Am livrat un **SCHELET COMPLET și FUNCȚIONAL** al întregului Sistem cu Inteligență Artificială (SIA), având toate cele 3 module (Achiziție, Model, UI) integrate și comunicând între ele.

---

## Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
| --- | --- | --- |
| Detectarea automată a defectelor critice (fisuri, pori) în suduri industriale | Clasificare imagine cu Rețea Neuronală Convoluțională (CNN) → rezultat "Defect/Bun" + grad încredere în < 1 secundă | `Neural Network` + `Web Service / UI` |
| Reducerea erorilor umane în inspecția vizuală (oboseala operatorului) | Sistem automat de validare ("Second Opinion") cu acuratețe țintă >95% | `Web Service / UI` (Afișare clară Verde/Roșu) |

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

### Contribuția originală la setul de date:

**Total observații finale:** **4116** (după Etapa 3 + Etapa 4)
**Observații originale:** **2744** (**66.67%**)

**Tipul contribuției:**
[X] Date generate prin simulare fizică (Augmentare sintetică avansată - Zgomot senzor & Iluminare)

[ ] Date achiziționate cu senzori proprii

[ ] Etichetare/adnotare manuală

[ ] Date sintetice prin metode avansate

**Descriere detaliată:**
Am implementat un script Python dedicat (`generate_data.py`) care triplează setul de date original. Pentru fiecare imagine reală, am generat programatic două variante sintetice care simulează condiții industriale dificile:

1. **Simulare Zgomot Senzor:** Adăugarea de zgomot Gaussian (Gaussian Noise) peste imaginea originală pentru a simula capturi cu senzori ISO ridicat sau interferențe.
2. **Variații de Iluminare:** Modificarea canalului 'Value' din spațiul HSV pentru a simula suduri fotografiate în condiții de întuneric sau supraexpunere (lumina arcului electric).

Aceste date forțează modelul să învețe trăsăturile structurale ale defectului (forma fisurii), nu doar culoarea sau luminozitatea pixelilor.

**Locația codului:** `src/data_acquisition/generate_data.py`
**Locația datelor:** `data/generated/` (și integrate în `data/train/`)

**Dovezi:**

* Raport generare terminal: *"Procent Contributie Proprie: 66.67%"*
* Folderele din `data/generated` conțin fișierele cu prefixul `gen_noise_` și `gen_aug_`.

---

### 3. Diagrama State Machine a Întregului Sistem

#### Diagrama Conceptuală a Fluxului

```
IDLE → LOAD_IMAGE (User Action) → PREPROCESS (Resize 224x224, Normalize) → 
RN_INFERENCE (CNN Model) → 
  ├─ [Score > Threshold] → DISPLAY_RESULT (Class + Confidence) → IDLE
  └─ [Error/Invalid] → DISPLAY_ERROR → IDLE

```

#### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **Clasificare la cerere (On-Demand Classification)**, deoarece proiectul vizează asistarea unui operator uman care verifică suduri specifice.

**Stările principale sunt:**

1. **IDLE:** Aplicația așteaptă input de la utilizator (stare pasivă).
2. **LOAD_IMAGE:** Încărcarea imaginii de pe disc și conversia în format numeric (array).
3. **PREPROCESS:** Pas critic unde imaginea este redimensionată la 224x224 și pixelii sunt normalizați la [0,1], exact ca la antrenare.
4. **RN_INFERENCE:** Modelul CNN calculează probabilitățile pentru cele 5 clase (`crack`, `porosity`, etc.).
5. **DISPLAY_RESULT:** Interpretarea vectorului de ieșire (argmax) și afișarea rezultatului color-coded (Verde=Bun, Roșu=Defect).

Starea **ERROR** gestionează cazurile în care fișierul încărcat nu este o imagine validă sau modelul nu este găsit.

---

### 4. Scheletul Complet al celor 3 Module

Toate cele 3 module sunt implementate în Python și sunt funcționale.

| **Modul** | **Tehnologie / Fișier** | **Stare Funcțională (Etapa 4)** |
| --- | --- | --- |
| **1. Data Acquisition** | `src/data_acquisition/generate_data.py` | **Finalizat.** Citește raw data, aplică zgomot/luminozitate și generează 4116 imagini organizate pe clase. |
| **2. Neural Network** | `src/neural_network/cnn_model.py` | **Finalizat.** Arhitectură CNN definită (3 blocuri convoluționale + Dropout), compilată și salvată. |
| **3. UI / App** | `src/app/gui_tf.py` (CustomTkinter) | **Finalizat.** Interfață grafică modernă care încarcă modelul `.keras`, permite upload de imagini și afișează predicția în timp real. |

#### Detalii per modul:

**Modul 1: Data Acquisition**

* Rulează cu: `python src/data_acquisition/generate_data.py`
* Output: Populează `data/train` cu datele originale + sintetice.
* Realizează automat split-ul și organizarea pe clase.

**Modul 2: Neural Network**

* Definit în clasa `WeldingCNN`.
* Include funcții de `_build_model` (arhitectură), `save_model` și `predict_image`.
* Antrenarea este gestionată de `src/neural_network/train.py` care folosește `image_dataset_from_directory`.

**Modul 3: Web Service / UI**

* Rulează cu: `python src/app/gui_tf.py`
* Construit cu biblioteca `customtkinter` pentru un aspect modern (Dark Mode).
* Funcționalitate completă: Buton Upload -> Preprocesare invizibilă -> Inferență -> Afișare text și culoare.

---

## Structura Repository-ului la Finalul Etapei 4

```
Sistem AI pentru Clasificarea Defectelor de Sudura/
├── data/
│   ├── raw/               # Datele brute (5 clase)
│   ├── train/             # Dataset complet (4116 imagini)
│   └── generated/         # Copie a datelor sintetice (dovada 40%)
├── src/
│   ├── data_acquisition/
│   │   ├── fix_dataset.py    # Curățare structură Roboflow
│   │   └── generate_data.py  # Modul 1: Generare date sintetice
│   ├── neural_network/
│   │   ├── cnn_model.py      # Modul 2: Definiție Arhitectură
│   │   ├── train.py          # Script antrenare
│   │   └── evaluate.py       # Script evaluare (Confusion Matrix)
│   └── app/
│       └── gui_tf.py         # Modul 3: Interfață Grafică
├── docs/
│   ├── state_machine.png
│   ├── confusion_matrix.png
│   └── screenshots/
│       └── inference_real.png
├── models/
│   └── trained_model.keras   # Modelul compilat și antrenat
├── README.md
├── README_Etapa3.md
└── README_Etapa4_Arhitectura_SIA.md  # ← Acest fișier

```

---

## Checklist Final

* [X] Tabelul Nevoie → Soluție completat
* [X] Declarație contribuție 66.67% date originale (zgomot + augmentare)
* [X] Diagrama State Machine explicată (IDLE -> PREPROCESS -> INFERENCE)
* [X] Modul 1 (Generare) funcțional
* [X] Modul 2 (CNN) definit și funcțional
* [X] Modul 3 (UI) funcțional și testat
* [X] Repository structurat corect