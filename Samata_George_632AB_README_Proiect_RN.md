# 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Samata George |
| **Grupa / Specializare** | 632AB / Stiinte ingineresti aplicate/ Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/GeorgeSamata/Sistem-AI-pentru-Clasificarea-Defectelor-de-Sudura |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow/Keras, CustomTkinter, NumPy, OpenCV) |
| **Domeniul Industrial de Interes (DII)** | Controlul Calității (QA) / Producție Metalurgică |
| **Tip Rețea Neuronală** | CNN (Convolutional Neural Network) - Arhitectură Custom |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 96.00% | 96.00% | +26.00% | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.96 | 0.96 | +0.31 | [✓] |
| Latență Inferență | <100 ms | 45 ms | 35 ms | -10 ms | [✓] |
| Contribuție Date Originale | ≥40% | 40% | 40% | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 | 4 | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, GitHub Copilot) a fost folosită ca unealtă de dezvoltare – pentru explicații concepte, debugging erori Python și structurarea documentației.

**Confirmare explicită:**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [X] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/augmentate/preprocesate de mine) | [X] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [X] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** | [X] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [X] DA     |

**Semnătură student:** Samata George

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria metalurgică și de construcții navale, inspecția sudurilor se realizează manual de către experți certificați. Acest proces este lent, subiectiv (depinde de oboseala inspectorului) și costisitor. O sudură defectă (fisură, porozitate) nedetectată poate duce la catastrofe structurale. Nevoia este de a automatiza trierea inițială a imaginilor radiografice sau macro pentru a asista operatorul uman.

### 2.2 Beneficii Măsurabile Urmărite

1. **Viteză:** Analiza unei imagini în sub 1 secundă (vs 1-2 minute manual).
2. **Consistență:** Eliminarea subiectivismului uman și a erorilor cauzate de oboseală.
3. **Reducerea Costurilor:** Filtrarea automată a pieselor evident defecte fără a bloca experții umani.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Detectarea fisurilor critice | Clasificare imagine cu CNN antrenat pe defecte structurale | Neural Network (Keras) | Recall > 90% pe clasa 'crack' |
| Alertarea rapidă a operatorului | Interfață grafică cu coduri de culoare (Roșu/Verde) | UI App (CustomTkinter) | Timp răspuns < 500ms |
| Auditarea inspecțiilor | Salvarea log-urilor cu predicții și grad de încredere | Data Logging (Python) | 100% istoric salvat |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | Mixt (Dataset public Roboflow + Augmentare Proprie) |
| **Sursa concretă** | Welding Defect Dataset (Kaggle/Roboflow) + Generare locală |
| **Număr total observații finale (N)** | ~5,500 imagini |
| **Număr features** | Imagine 224x224x1 (Grayscale) |
| **Tipuri de date** | Imagini (JPG/PNG) |
| **Clase** | 5 (Bad Weld, Crack, Good Weld, Porosity, Spatter) |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 5,500 |
| **Observații originale/augmentate (M)** | 2,200 |
| **Procent contribuție originală** | 40% |
| **Tip contribuție** | Augmentare avansată (Flip, Rotation, Noise, Brightness) + Preprocesare Grayscale |
| **Locație cod generare** | `src/data_acquisition/generate_data.py` |

**Descriere metodă:**
Am preluat un set de bază și am aplicat un pipeline propriu de augmentare pentru a echilibra clasele (clasa 'crack' era sub-reprezentată). Am generat variații sintetice folosind librăria `ImageDataGenerator` din Keras, simulând condiții diferite de iluminare și poziționare a piesei, specifice mediului industrial real.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | ~3,850 |
| Validation | 20% | ~1,100 |
| Test | 10% | ~550 |

**Preprocesări aplicate:**
- **Conversie Grayscale:** Eliminarea zgomotului de culoare (metalul are culori irelevante pentru defect).
- **Resize:** Standardizare la 224x224 pixeli.
- **Normalizare:** Scalarea pixelilor la intervalul [0, 1] (împărțire la 255).

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging** | Python (OS/Pandas) | Încărcare date, organizare foldere, salvare istoric | `src/data_acquisition/` |
| **Neural Network** | TensorFlow/Keras | Model CNN pentru clasificarea defectelor | `src/neural_network/` |
| **UI Application** | CustomTkinter | Interfață Desktop pentru operatori (Upload/Analyze) | `src/app/gui_tf.py` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale:**
1.  **IDLE:** Așteptare încărcare imagine.
2.  **PREPROCESS:** Conversie Grayscale și Resize la 224x224.
3.  **INFERENCE:** Modelul CNN calculează probabilitățile.
4.  **DECISION:** Aplicare `argmax` și verificare prag încredere.
5.  **ALERT:** Afișare rezultat (Verde=Conform, Roșu=Neconform).

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale (CNN Custom)

Input (224, 224, 1)
  → Conv2D(32, 3x3) + ReLU → MaxPool(2x2)
  → Conv2D(64, 3x3) + ReLU → MaxPool(2x2)
  → Conv2D(128, 3x3) + ReLU → MaxPool(2x2)
  → Flatten
  → Dense(128) + ReLU → Dropout(0.5)
  → Dense(5) + Softmax (Output)

**Justificare:** Am ales o arhitectură secvențială clasică (tip VGG simplificat) deoarece defectele de sudură (fisuri, pori) sunt elemente vizuale locale (margini, texturi) pe care straturile de convoluție le extrag eficient.

### 5.2 Hiperparametri Finali

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Optimizer | Adam | Convergență rapidă și adaptivă |
| Learning Rate | 0.001 | Valoare standard pentru start stabil |
| Batch Size | 32 | Balans optim între viteză și stabilitate gradient |
| Epochs | 21 (din max 30) | Oprit automat de Early Stopping |
| Loss Function | Sparse Cat. Crossentropy | Clasificare multi-clasă cu etichete integer |

### 5.3 Experimente de Optimizare

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Observații |
|------|----------------------------|----------|----------|------------|
| **Baseline** | CNN simplu, 10 epoci | 75% | 0.72 | Underfitting |
| Exp 1 | + Augmentare Date | 85% | 0.82 | Generalizare mai bună |
| Exp 2 | + Dropout 0.5 | 91% | 0.89 | Reducere Overfitting |
| **FINAL** | **+ Early Stopping (Patience=5)** | **96%** | **0.96** | **Model Optim** |

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | **96.00%** | ≥70% | [✓] |
| **F1-Score** | **0.96** | ≥0.65 | [✓] |

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
- **Puncte tari:** Modelul distinge perfect între `Good Weld` și `Crack` (erori zero pe defecte critice).
- **Confuzii:** Există o ușoară confuzie între `Bad Weld` (neconformitate generală) și `Spatter` (stropi), deoarece vizual sunt similare (ambele prezintă neregularități de suprafață).

### 6.3 Analiza Erori

| # | Input | Predicție | Real | Cauză Probabilă |
|---|-------|-----------|------|-----------------|
| 1 | Imagine Google (Color) | Bad Weld | Good Weld | **Domain Shift:** Iluminare diferită față de training set |
| 2 | Sudură cu reflexii | Crack | Good Weld | Reflexia metalului interpretată ca fisură albă |

**Implicație:** Sistemul trebuie utilizat în condiții de iluminare controlată, similar cu datele de antrenament.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Modificare Etapa 6 | Justificare |
|------------|-------------------|-------------|
| **Model** | Încărcare `optimized_model.keras` | Performanță maximă (96%) |
| **Terminologie** | Traducere ISO (ex: "Fisură") | Limbaj tehnic pentru operatori români |
| **UI** | Afișare procent Încredere | Operatorul poate decide manual dacă AI-ul e nesigur |
| **Siguranță** | Try-Catch pe încărcare fișier | Prevenire crash aplicație la erori I/O |

### 7.2 Screenshot UI

**Locație:** `docs/screenshots/inference_optimized.png`
Demonstrează interfața CustomTkinter rulând predicția "DEFECT STRUCTURAL - Fisură" pe o imagine de test.

---

## 8. Structura Repository-ului Final

proiect-rn-Samata-George/
│
├── README.md                               # Documentație Finală
├── docs/
│   ├── confusion_matrix_optimized.png      # Matrice confuzie
│   ├── loss_curve.png                      # Grafic antrenare
│   └── screenshots/
│       └── inference_optimized.png         # Screenshot UI
├── data/
│   ├── train/                              # Date antrenare
│   └── test/                               # Date testare
├── src/
│   ├── data_acquisition/generate_data.py   # Script augmentare
│   ├── neural_network/
│   │   ├── train.py                        # Script antrenare
│   │   ├── cnn_model.py                    # Arhitectură
│   │   └── generate_matrix.py              # Generare grafice
│   └── app/
│       └── gui_tf.py                       # APLICATIA FINALĂ (UI)
├── models/
│   ├── optimized_model.keras               # MODEL FINAL (v0.6)
│   └── trained_model.keras                 # Model vechi
└── requirements.txt                        # Dependențe

---

## 9. Instrucțiuni de Rulare

1.  **Clonare și instalare:**
    ```bash
    git clone [https://github.com/GeorgeSamata/Sistem-AI-pentru-Clasificarea-Defectelor-de-Sudura.git](https://github.com/GeorgeSamata/Sistem-AI-pentru-Clasificarea-Defectelor-de-Sudura.git)
    pip install -r requirements.txt
    ```

2.  **Lansare Aplicație (UI):**
    ```bash
    python src/app/gui_tf.py
    ```
    *Se va deschide fereastra aplicației. Încărcați o imagine din folderul `data/test` sau `demo_images`.*

3.  **Regenerare Grafice (Opțional):**
    ```bash
    python src/neural_network/generate_matrix.py
    ```

---

## 10. Concluzii și Lecții Învățate

### 10.1 Evaluare
Proiectul a depășit target-ul de 70% acuratețe, atingând **96%**. Modelul este robust pe date similare celor de antrenament și integrează mecanisme de siguranță (Early Stopping).

### 10.2 Limitări
Limitarea principală este **generalizarea pe imagini de pe internet** care au condiții de iluminare drastic diferite. Modelul este calibrat pentru inspecție industrială standardizată (unghi fix, lumină constantă).

### 10.3 Lecții Învățate
1.  **Datele > Modelul:** Preprocesarea (Grayscale) a avut un impact mai mare decât adăugarea de straturi noi în CNN.
2.  **Early Stopping:** Este esențial pentru a economisi timp și a preveni overfitting-ul.
3.  **UX Industrial:** Operatorii au nevoie de termeni clari ("Conform/Neconform"), nu de probabilități matematice brute.

---

## 11. Bibliografie

1. TensorFlow Documentation, 2024. *Image Classification with Keras*. URL: https://www.tensorflow.org/tutorials/images/classification
2. Kaggle, 2023. *Welding Defect Dataset*. URL: https://www.kaggle.com/datasets
3. Roboflow Universe, 2024. *Computer Vision Datasets*. URL: https://universe.roboflow.com/

---

## 12. Checklist Final

- [X] **Accuracy ≥70%** (Realizat: 96%)
- [X] **F1-Score ≥0.65** (Realizat: 0.96)
- [X] **Contribuție ≥40% date originale** (Augmentare + Procesare)
- [X] **Model antrenat de la zero**
- [X] **Minimum 4 experimente**
- [X] **Confusion matrix** prezentă
- [X] **Cele 3 module funcționale** (Data, RN, UI)
- [X] **Tag Git `v0.6-optimized-final`**

**Versiune document:** FINAL pentru examen
**Data:** 10.02.2026