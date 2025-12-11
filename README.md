# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Șamata George Cristian  
**Data:** 13.12.2025 

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Setul de date este compus din trei surse:
    1. Imagini publice de sudură (dataset Kaggle) cu defecte precum fisuri și porozitate.
    2. Imagini originale, obținute prin fotografierea unor suduri reale.
    3. Imagini generate programatic prin augmentare și simulare de defecte.
* **Modul de achiziție:** [x] Senzori reali (cameră) / [x] Simulare / [x] Fișier extern / [x] Generare programatică
* **Perioada / condițiile colectării:** Noiembrie 2024 – Ianuarie 2025. Imaginile brute au fost capturate cu un telefon mobil în condiții variabile de lumină ambientală. Imaginile sintetice au fost generate în Python.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** Aproximativ 1.200 imagini.
* **Număr de caracteristici (features)::** 3 caracteristici numerice extrase + 1 etichetă.
* **Tipuri de date:** [ ] Numerice / [ ] Categoriale / [ ] Temporale / [x] Imagini (cu extracție ulterioară de features numerice)
* **Format fișiere:** [x] CSV / [ ] TXT / [ ] JSON / [x] PNG/JPG / [ ] Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

Modelul primește trei intrări numerice, obținute prin procesarea fiecărei imagini de sudură (feature extraction), și o etichetă de clasă.

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| mean_intensity | numeric | nivel pixel | Media intensității pixelilor (evaluare expunere) | 0–255 |
| edge_density | numeric | procent | Raportul pixelilor detectați ca margini (Canny) | 0–1 |
| texture_roughness | numeric | u.a. | Variabilitatea texturii (varianța Laplacianului) | 0–∞ |
| label | categorial | - | Eticheta: OK, CRACK (fisură), POROSITY (porozitate) | {OK, CRACK, POROSITY} |


---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distribuția rezoluției** imaginilor brute.
* **Histograme ale intensității pixelilor** (brightness) pentru verificarea expunerii.
* **Histograme pentru caracteristicile extrase:** `edge_density` și `texture_roughness`.
* **Raportul de distribuție pe clase** (OK vs. CRACK vs. POROSITY).
* **Identificarea zgomotului:** detectarea imaginilor cu zgomot excesiv sau expunere neuniformă.

### 3.2 Analiza calității datelor

* **Variații de iluminare:** Detectate în imaginile brute, necesitând normalizare.
* **Dezechilibru de clasă:** Inițial existau mai multe imagini OK decât defecte (tratat ulterior prin augmentare).
* **Rezoluție:** O parte din imaginile raw aveau rezoluție prea mică și au fost eliminate.
* **Valori lipsă:** Nu există (imaginile sunt procesate programatic).

### 3.3 Probleme identificate

* Dezechilibru moderat între clase.
* Diferențe vizuale mari între datasetul public (Kaggle) și imaginile originale (fotografiate).
* Necesitatea normalizării intensității pentru a reduce influența condițiilor de iluminare ambientală.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminarea duplicatelor.**
* **Filtrare:** Eliminarea imaginilor neclare sau cu rezoluție insuficientă.
* **Conversie:** Uniformizare în format PNG/JPG.

### 4.2 Transformarea caracteristicilor

* **Redimensionare:** Toate imaginile aduse la 224x224 pixeli.
* **Normalizare:** Scalarea valorilor pixelilor în intervalul [0, 1].
* **Feature Extraction:** Calcularea `mean_intensity`, `edge_density`, `texture_roughness`.
* **Augmentare:**
  * Ajustare lumină și contrast.
  * Adăugare zgomot gaussian.
  * Blur pentru simularea vibrațiilor.
  * Generare texturi artificiale pentru simularea porozității.

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70% – train
* 15% – validation
* 15% – test

**Principii respectate:**
* Stratificare pe clase pentru a păstra proporțiile în toate seturile.
* Fără scurgere de informație (data leakage).
* Statisticile pentru normalizare au fost calculate **DOAR** pe setul de train.

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate (imagini normalizate) în `data/processed/`.
* Seturi train/val/test în folderele dedicate.
* Parametrii de preprocesare salvați în `config/preprocessing_config.txt`.

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute (imagini publice + originale)
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea detaliată a dataset-ului

---

##6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README 

---
```