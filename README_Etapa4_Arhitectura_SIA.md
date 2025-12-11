# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Șamata George Cristian  
**Data:** 11.12.2025  
---



### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Inspecția lentă a sudurilor industriale (manual durează 2-5 min/piesă) | Clasificare automată (OK vs Defect) în timp real (< 100ms/imagine) | `src/neural_network` (CNN TensorFlow) |
| Riscul omiterii fisurilor fine din cauza oboselii operatorului | Analiză obiectivă pixel-cu-pixel și alertare vizuală imediată în interfață | `src/app` (Interfață Streamlit) |
| Lipsa datelor variate pentru antrenarea modelelor robuste | Generare sintetică a 1000+ imagini cu zgomot și rotații pentru balansare clase | `src/data_acquisition` (Data Augmentation) |

**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

#### Cum se calculează 40%:
Am pornit de la dataset-ul public Kaggle (~1500 imagini). Pentru a atinge cerința de 40%, am generat sintetic un set suplimentar de date.

#### Declarație obligatorie în README:

### Contribuția originală la setul de date:

**Total observații finale:** ~2,500 imagini (după Etapa 3 + Etapa 4)
**Observații originale:** ~1,000 imagini (40%)

**Tipul contribuției:**
[ ] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[X] Etichetare/adnotare manuală  
[X] Date sintetice prin metode avansate  

**Descriere detaliată:**
Deoarece dataset-ul original Kaggle este limitat și dezechilibrat, am implementat un modul propriu de **Data Augmentation** (`src/data_acquisition/generate_tf_data.py`). Acesta aplică transformări realiste imaginilor existente pentru a simula condiții diverse din fabrică:
1.  **Zgomot Gaussian:** Simularea senzorilor cu ISO ridicat în condiții de lumină slabă.
2.  **Transformări Geometrice:** Flip orizontal/vertical și rotații random, deoarece poziția piesei pe bandă poate varia.
3.  **Variații de Iluminare:** Modificarea canalului Value (din HSV) pentru a simula suduri supraexpuse sau subexpuse.

Aceste date sunt generate fizic în folderul `data/generated` și sunt folosite pentru antrenare alături de cele raw.

**Locația codului:** `src/data_acquisition/generate_tf_data.py`
**Locația datelor:** `data/generated/`

**Dovezi:**
- Grafic comparativ: `docs/data_statistics.csv`
- Folderul `data/generated` conține fișierele cu prefixul `aug_`.

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.png`
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Diagramă Textuală (reprezentare a imaginii din docs/):**
```

IDLE → UPLOAD\_IMAGE (User UI) → RESIZE\_AND\_NORMALIZE (Tensor 224x224x3) →
├─ [Preprocessing Error] → LOG\_ERROR → SHOW\_MSG → IDLE
└─ [Valid] → LOAD\_TF\_MODEL (.keras) →
CNN\_FORWARD\_PASS (Conv2D -\> ReLU -\> MaxPool) →
SOFTMAX\_PROBABILITY → DECISION\_LOGIC (\> 0.5) →
├─ [Class: OK] → SHOW\_GREEN\_BOX → LOG\_ENTRY → IDLE
└─ [Class: DEFECT] → SHOW\_RED\_ALERT → LOG\_ENTRY → IDLE
↓ [User Exit]
STOP\_SYSTEM

```

**Legendă obligatorie (scrieți în README):**

### Justificarea State Machine-ului ales:

Am ales arhitectura de tip **Clasificare la cerere (Trigger-based)** pentru că proiectul nostru vizează asistarea unui operator uman care încarcă radiografii punctuale pentru verificare (inspecție off-line).

Stările principale sunt:
1.  **RESIZE_AND_NORMALIZE:** Stare critică pentru rețelele neuronale (CNN). Imaginile brute au rezoluții diverse, dar modelul TensorFlow acceptă doar tensori de dimensiune fixă (224x224) cu valori normalizate [0,1].
2.  **CNN_FORWARD_PASS:** Execuția efectivă a modelului (inferența), unde imaginea trece prin straturile convoluționale.
3.  **DECISION_LOGIC:** Interpretarea vectorului de probabilități (Softmax). Deși rețeaua dă un procent (ex: 0.85 Defect), sistemul trebuie să ia o decizie binară clară pentru operator (OK/NOT OK) bazată pe un prag de siguranță.

Tranzițiile critice sunt gestionate prin verificări de eroare (ex: dacă imaginea nu poate fi citită, se trece în starea LOG_ERROR fără a bloca aplicația).

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | **MUST:** Produce CSV/Imagini cu datele voastre (inclusiv cele 40% originale). Codul generează datele sintetice în `data/generated`. |
| **2. Neural Network Module** | `src/neural_network/cnn_model.py` | **MUST:** Modelul RN definit (CNN), compilat, poate fi încărcat și salvat ca `.keras`. |
| **3. Web Service / UI** | Streamlit (`src/app/gui_tf.py`) | **MUST:** Primește input de la user și afișează un output (Clasa OK/Defect). |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [X] Cod rulează fără erori: `python src/data_acquisition/generate_tf_data.py`
- [X] Generează date sintetice (imagini augmentate) în `data/generated`
- [X] Include minimum 40% date originale în dataset-ul final prin augmentare
- [X] Documentație în cod: pipeline-ul de augmentare TensorFlow/OpenCV

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [X] Arhitectură RN definită și compilată fără erori (CNN Secvențial cu straturi Conv2D și Dense)
- [X] Model poate fi salvat (`save_model`) și reîncărcat
- [X] Include justificare pentru arhitectura aleasă (în docstring)
- [X] **NU trebuie antrenat** cu performanță bună (weights sunt inițializate random/default pentru schelet)

#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [X] Propunere Interfață ce primește input de la user (file upload)
- [X] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```

Sistem-AI-pentru-Clasificarea-Defectelor-de-Sudura/
├── data/
│   ├── raw/                  \# Datele originale Kaggle
│   ├── generated/            \# Date originale (Sintetice - Contribuția mea)
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data\_acquisition/
│   │   ├── generate\_tf\_data.py \# MODUL 1
│   │   └── README.md
│   ├── preprocessing/
│   ├── neural\_network/
│   │   ├── cnn\_model.py      \# MODUL 2
│   │   ├── train\_model.py
│   │   └── README.md
│   └── app/                  \# MODUL 3
│       ├── gui\_tf.py
│       └── README.md
├── docs/
│   ├── state\_machine.png     \# OBLIGATORIU
│   └── screenshots/
├── models/
│   └── welding\_model\_v1.keras \# Modelul salvat
├── config/
├── README.md
├── README\_Etapa3.md              \# (deja existent)
├── README\_Etapa4\_Arhitectura\_SIA.md              \# ← acest fișier completat (în rădăcină)
└── requirements.txt

```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [X] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [X] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [X] Cod generare/achiziție date funcțional și documentat
- [X] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [X] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [X] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [X] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [X] Cod rulează fără erori (`python src/data_acquisition/generate_tf_data.py`)
- [X] Produce minimum 40% date originale din dataset-ul final
- [X] CSV/Imagini generate în format compatibil cu preprocesarea
- [X] Documentație în `src/data_acquisition/README.md` cu metoda explicată
- [X] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [X] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [X] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [X] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [X] Screenshot demonstrativ în `docs/screenshots/`
- [X] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:** `"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:** `git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`
```