# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale

**Instituție:** POLITEHNICA București – FIIR

**Student:** Șamata George Cristian
**Link Repository GitHub:** [Adaugă Linkul Aici]
**Data predării:** 22.01.2026

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN**.
Am antrenat modelul CNN definit în Etapa 4 pe setul de date complet (4116 imagini, din care 66.67% originale) și am obținut o performanță de nivel industrial.

---

## PREREQUISITE – Verificare Etapa 4

* [X] **State Machine** definit și documentat în `docs/state_machine.png`.
* [X] **Contribuție ≥40% date originale:** 2744 imagini (66.67%) generate și stocate în `data/generated`.
* [X] **Modul 1 (Data Logging):** Script `generate_data.py` funcțional.
* [X] **Modul 2 (RN):** Arhitectură CNN definită în `cnn_model.py`.
* [X] **Modul 3 (UI):** Interfață funcțională (`gui_tf.py`) care încarcă modelul.

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu

1. **Antrenare model:** Modelul a fost antrenat pe 30 epoci.
2. **Împărțire stratificată:** 80% Train / 20% Validare (automat prin `image_dataset_from_directory`).
3. **Metrici test:**
* **Acuratețe:** **96% (0.96)** (Target: ≥ 65%)
* **F1-score (macro):** **0.95** (Target: ≥ 0.60)


4. **Model salvat:** `models/trained_model.keras`.

#### Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
| --- | --- | --- |
| **Learning Rate** | 0.001 | Valoare standard pentru optimizatorul Adam; oferă convergență rapidă fără instabilitate. |
| **Batch Size** | 32 | Compromis optim între viteza de procesare și stabilitatea gradientului pe dataset-ul de ~4000 imagini. |
| **Epochs** | 30 | Suficiente pentru ca modelul să conveargă (s-a stabilizat la epoca 6 cu acc=95%). |
| **Optimizer** | Adam | Algoritm adaptiv, ideal pentru CNN-uri, gestionează automat learning rate-ul per parametru. |
| **Loss Function** | Categorical Crossentropy | Problema este multi-class (5 clase disjuncte: `bad_weld`, `crack`, etc.), deci aceasta este funcția corectă. |
| **Architecture** | 3 x Conv2D+MaxPool | Arhitectură clasică VGG-like: extrage trăsături ierarhice (margini -> forme -> defecte complexe). |
| **Dropout** | 0.5 | Aplicat înainte de stratul Dense final pentru a preveni overfitting-ul (forțează redundanța neuronilor). |

---

### Nivel 2 – Recomandat

1. **Early Stopping:** Configurat cu `patience=5` (oprește antrenarea dacă `val_loss` nu scade timp de 5 epoci).
2. **Augmentări:** Zgomot Gaussian și Variații HSV aplicate în Etapa 4 (offline augmentation).
3. **Grafic Loss:** Salvat în `docs/loss_curve.png`.
4. **Analiză Erori:** Detaliată mai jos.

**Indicatori obținuți:**

* **Acuratețe: 96%** (Peste pragul de 75% pentru Nivel 2)
* **F1-score: 0.95** (Peste pragul de 0.70 pentru Nivel 2)

---

### Nivel 3 – Bonus

1. **Confusion Matrix:** Generată și salvată în `docs/confusion_matrix.png`.
2. **Analiză detaliată:** Modelul are **Recall 1.00 (Perfect)** pe clasa critică `crack`.

---

## Analiză Erori în Context Industrial

### 1. Pe ce clase greșește cel mai mult modelul?

Conform Raportului de Clasificare (`results/final_metrics.txt`):

* **Cea mai "slabă" clasă:** `good_weld` (F1-score 0.87).
* **Confuzie:** Modelul tinde să clasifice unele suduri bune (`good_weld`) drept `bad_weld` (False Positive).

### 2. Ce caracteristici ale datelor cauzează erori?

* Clasa `good_weld` are trăsături vizuale foarte subtile care o diferențiază de `bad_weld` (uneori doar neregularități minore de suprafață).
* Iluminarea variabilă din datele augmentate poate crea umbre pe o sudură bună, pe care modelul le interpretează ca fiind defecte de formă (`bad_weld`).

### 3. Ce implicații are pentru aplicația industrială?

Aceasta este o situație **IDEALĂ** pentru siguranță:

* **False Negatives (Defect ratat):** Aproape ZERO. (Recall pe `crack` este 1.00, pe `spatter` 0.99).
* **False Positives (Alarmă falsă):** Existente.
* **Impact:** Sistemul este "prudent". Mai bine respinge o piesă bună (care va fi reverificată manual și validată) decât să lase să treacă o piesă cu fisură (`crack`) care ar putea duce la cedarea structurii.

### 4. Ce măsuri corective propuneți?

1. **Colectare date specifice:** Adăugarea a încă 200 de imagini cu `good_weld` în condiții de iluminare dificilă pentru a învăța modelul să ignore umbrele.
2. **Fine-tuning Threshold:** Creșterea pragului de încredere pentru clasa `bad_weld` (ex: să raporteze defect doar dacă e 90% sigur).
3. **Adăugare clasă "Ambiguu":** Dacă încrederea e sub 60%, sistemul să ceară explicit intervenția operatorului.

---

## Structura Repository-ului la Finalul Etapei 5

```
Sistem AI pentru Clasificarea Defectelor de Sudura/
├── README_Etapa5_Antrenare_RN.md  # ← ACEST FIȘIER
├── docs/
│   ├── loss_curve.png                 # Grafic antrenare (Train vs Val)
│   ├── confusion_matrix.png           # Matricea de confuzie
│   └── screenshots/
│       └── inference_real.png         # Screenshot UI cu predicție reală
├── results/
│   ├── training_history.csv           # Log-ul complet pe 30 epoci
│   └── final_metrics.txt              # Raport detaliat (Precision/Recall)
├── models/
│   └── trained_model.keras            # Modelul antrenat (SALVAT)
├── src/
│   ├── neural_network/
│   │   ├── train.py                   # Script antrenare
│   │   ├── evaluate.py                # Script generare matrice/raport
│   │   └── cnn_model.py               # Arhitectură
│   └── app/
│       └── gui_tf.py                  # UI actualizat
└── data/                              # Dataset-ul complet

```

---

## Instrucțiuni de Rulare

### 1. Antrenare Model (Reproducere rezultate)

```bash
python src/neural_network/train.py

```

*Output:* Va rula 30 epoci și va salva modelul în `models/trained_model.keras` (cu acuratețe ~96%).

### 2. Generare Rapoarte (Matrice Confuzie)

```bash
python src/neural_network/evaluate.py

```

*Output:* Generează `docs/confusion_matrix.png` și `results/final_metrics.txt`.

### 3. Testare UI cu Model Antrenat

```bash
python src/app/gui_tf.py

```

1. Apasă "Incarca Imagine".
2. Selectează o imagine din `data/raw/crack` (sau alt folder).
3. Apasă "Analizeaza".
4. Rezultatul va fi afișat instant (ex: `CRACK - 99.8%`).

---

## Checklist Final – Bifat pentru Predare

* [X] Model antrenat de la zero (30 epoci).
* [X] Acuratețe Test 96% (Mult peste pragul de 65%).
* [X] Tabel hiperparametri completat și justificat.
* [X] Screenshot inferență reală în UI (`docs/screenshots/inference_real.png`).
* [X] Analiză erori în context industrial realizată.
* [X] Grafic Loss și Matrice de Confuzie generate.

**Concluzie:** Modelul este robust, sigur pentru aplicații industriale (Recall mare pe defecte critice) și integrat complet în aplicație.