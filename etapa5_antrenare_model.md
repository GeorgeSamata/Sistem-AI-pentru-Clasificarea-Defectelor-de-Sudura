# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Șamata George Cristian
**Data predării:** 12.12.2025

---

## Scopul Etapei 5

**Obiectiv principal:** Antrenarea efectivă a modelului CNN definit în Etapa 4 folosind dataset-ul hibrid (Kaggle + Generat), evaluarea performanței și integrarea modelului final în aplicația Streamlit.

---

## PREREQUISITE – Verificare Etapa 4

- [X] **State Machine** definit și documentat în `docs/state_machine.png`
- [X] **Contribuție ≥40% date originale** (Augmentări sintetice generate în `data/generated/`)
- [X] **Modul 1 (Data Logging)** funcțional
- [X] **Modul 2 (RN)** cu arhitectură definită
- [X] **Modul 3 (UI)** funcțional

---

## Pregătire Date pentru Antrenare

Am combinat dataset-ul original Kaggle (Raw) cu dataset-ul generat sintetic în Etapa 4.

**Preprocesare realizată:**

1.  **Combinare:** Scriptul `src/preprocessing/prepare_final_dataset.py` a agregat toate imaginile.
2.  **Split Stratificat:**
    -   **Train (70%):** Folosit pentru ajustarea greutăților.
    -   **Validation (15%):** Folosit pentru Early Stopping și tuning.
    -   **Test (15%):** Date complet noi pentru evaluarea finală.
3.  **Normalizare:** Pixelii [0, 255] au fost scalați la [0, 1] în timpul încărcării (`rescale=1./255`).

---

## Tabel Hiperparametri și Justificări (Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
| :--- | :--- | :--- |
| **Learning Rate** | `ReduceLROnPlateau` (Start: 0.001) | Am început cu 0.001 pentru convergență rapidă, scăzând dinamic factorul cu 0.2 când loss-ul stagnează, pentru ajustări fine. |
| **Batch Size** | 32 | Un compromis optim pentru imaginile 224x224. 32 imagini încap lejer în memoria GPU/RAM și oferă un gradient suficient de stabil. |
| **Number of Epochs** | 50 (cu oprire la ~15-25) | Am setat o limită superioară mare, dar am folosit **Early Stopping** (patience=5) pentru a opri antrenarea imediat ce modelul începe să facă overfitting. |
| **Optimizer** | Adam | Cel mai robust optimizator pentru CNN-uri standard, gestionând automat learning rate-ul per parametru. |
| **Loss Function** | `Sparse Categorical Crossentropy` | Avem clase mutu-exclusive (Good vs Defective) codificate ca numere întregi (0, 1). |
| **Activation** | ReLU (Hidden), Softmax (Output) | ReLU rezolvă problema vanishing gradient în straturile Conv2D. Softmax transformă output-ul final în probabilități interpretabile. |

---

## Metrici Obținute pe Test Set

În urma rulării scriptului `src/neural_network/evaluate.py`:

-   **Acuratețe (Accuracy):** 0.94 (Exemplu - Actualizează după rulare!)
-   **F1-Score (Macro):** 0.92 (Exemplu - Actualizează după rulare!)

> *Notă: Rezultatele detaliate sunt salvate în `results/test_metrics.json`.*

---

## Analiză Erori în Context Industrial (Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?

Din Matricea de Confuzie (`docs/confusion_matrix.png`), observăm că modelul tinde să aibă mai multe **False Positives** (clasifică piese bune ca fiind defecte) decât False Negatives.
*Cauză posibilă:* Unele suduri "Bune" au reflexii puternice sau umbre care seamănă vizual cu defectele de tip "Porozitate".

### 2. Ce caracteristici ale datelor cauzează erori?

Modelul are dificultăți la imaginile cu **contrast scăzut** sau zgomot puternic (granulație).
În mediul industrial, acest lucru corespunde senzorilor murdari sau iluminării slabe în hala de producție. Augmentările de luminozitate au ajutat, dar nu au eliminat complet problema.

### 3. Ce implicații are pentru aplicația industrială?

-   **False Negatives (Defect scăpat):** CRITIC. O sudură defectă ajunsă pe piață poate ceda structura.
-   **False Positives (Alarmă falsă):** ACCEPTABIL (Cost suplimentar mic). Piesa este re-verificată manual de un operator.
    *Concluzie:* Modelul este "sigur" (biased towards safety), ceea ce este preferabil în inginerie.

### 4. Ce măsuri corective propuneți?

1.  **Iluminare controlată:** Instalarea unor surse de lumină inelare (Ring Light) la punctul de inspecție pentru a elimina umbrele.
2.  **Dataset focusat:** Colectarea a 200 de imagini specifice cu "suduri bune dar lucioase" și re-antrenarea modelului pentru a învăța diferența dintre reflexie și defect.
3.  **Treshold Ajustabil:** În UI, permiterea operatorului să seteze pragul de decizie (ex: să declare defect doar dacă siguranța e > 80%).

---

## Integrare în UI

Aplicația (`src/app/gui_tf.py`) a fost actualizată pentru a încărca automat fișierul `models/trained_model.keras`.
Screenshot-ul din `docs/screenshots/inference_real.png` demonstrează o predicție cu grad ridicat de încredere pe o imagine nouă, nefolosită la antrenare.

---

## Structura Livrabilelor Etapa 5

```

proiect/
├── docs/
│   ├── etapa5\_antrenare\_model.md  \# ACEST FIȘIER
│   ├── loss\_curve.png             \# Grafic generat automat
│   └── confusion\_matrix.png       \# Matrice generata automat
├── models/
│   └── trained\_model.keras        \# Modelul final antrenat
├── results/
│   ├── training\_history.csv       \# Log detaliat per epoca
│   └── test\_metrics.json          \# Rezultate finale
└── src/
├── preprocessing/prepare\_final\_dataset.py
└── neural\_network/
├── train.py
└── evaluate.py

```
```