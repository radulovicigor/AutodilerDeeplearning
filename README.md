# Auto Diler AI - Predikcija Cijena Automobila sa Dubokim Učenjem

Pametna aplikacija za predikciju cijena automobila korištenjem dubokog učenja i mašinskog učenja.

## 📋 Opis Projekta

Auto Diler AI je web aplikacija koja demonstrira primjenu tehnika dubokog učenja na tabelarnim podacima. Aplikacija omogućava:

- **Predikciju cijena automobila** (regresija) na osnovu karakteristika vozila
- **Klasifikaciju cjenovnih segmenata** (budget/mid/premium)
- **Interaktivno kreiranje neuronskih mreža** sa vizualizacijom arhitekture
- **Poređenje različitih ML modela** (Linear Regression, Random Forest, XGBoost, MLP)
- **Eksperimentisanje sa hiperparametrima** i praćenje rezultata

## 🛠️ Tehnologije

### Backend
- **FastAPI** - REST API framework
- **PyTorch** - Deep learning framework
- **scikit-learn** - Mašinsko učenje
- **SQLite + SQLModel** - Baza podataka
- **pandas, numpy** - Obrada podataka
- **matplotlib, seaborn** - Vizualizacije

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Styling
- **Recharts** - Grafovi
- **Framer Motion** - Animacije

## 📊 Dataset

Koristi se dataset sa **~7000 oglasa automobila** sa karakteristikama:
- **Kategorijalne:** marka, model, oštećenje, registracija, gorivo, mjenjač
- **Numeričke:** snaga (HP), kilometraža, kubikaža, godina
- **Target:** cijena (€)

## 🚀 Instalacija i Pokretanje

### Preduslovi
- Python 3.10+
- Node.js 18+
- CUDA toolkit (opciono, za GPU akceleraciju)

### Backend Setup

```bash
cd backend

# Kreiranje virtualnog okruženja
python -m venv venv

# Aktivacija (Windows)
venv\Scripts\activate

# Aktivacija (Linux/Mac)
source venv/bin/activate

# Instalacija zavisnosti
pip install -r requirements.txt

# Pokretanje servera
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Instalacija zavisnosti
npm install

# Pokretanje dev servera
npm run dev
```

### Brzo Pokretanje (Windows)

Koristi `POKRENI.bat` za automatsko pokretanje oba servera.

## 📁 Struktura Projekta

```
Auto-Diler-AI/
├── backend/
│   ├── app/
│   │   ├── config.py          # Konfiguracija
│   │   ├── database.py        # SQLModel modeli
│   │   ├── data_processing.py # ETL i preprocessing
│   │   ├── schemas.py         # Pydantic šeme
│   │   ├── training_service.py# Servis za trening
│   │   ├── prediction_service.py
│   │   ├── visualization.py   # Matplotlib plotovi
│   │   └── models/
│   │       ├── sklearn_models.py  # Linear, RF, XGBoost
│   │       └── pytorch_models.py  # MLP Regressor/Classifier
│   ├── main.py               # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Home
│   │   │   ├── predict/page.tsx  # Predikcija
│   │   │   ├── compare/page.tsx  # Poređenje modela
│   │   │   └── network-lab/page.tsx # MLP builder
│   │   ├── components/
│   │   │   ├── ui/              # UI komponente
│   │   │   └── NetworkVisualization.tsx
│   │   └── lib/
│   │       ├── api.ts           # API klijent
│   │       └── utils.ts
│   └── package.json
├── data/
│   └── autici_7k.csv           # Dataset
├── models/                      # Sačuvani modeli
├── experiments/                 # Artefakti eksperimenata
└── README.md
```

## 🎯 Funkcionalnosti

### 1. Predict Page
- Unos karakteristika vozila
- Odabir modela za predikciju
- Prikaz predviđene cijene sa objašnjenjem

### 2. Model Comparison
- Tabela svih treniranih modela
- Metrike: R², MAE, RMSE (regresija) / Accuracy, F1 (klasifikacija)
- **Side-by-Side poređenje** dva modela
- Vizualizacije: scatter plot, residuals, confusion matrix, feature importance
- Brisanje i preimenovanje modela

### 3. Network Lab
- Interaktivni builder MLP mreže
- Konfigurisanje slojeva, neurona, aktivacija
- **Real-time prikaz broja parametara**
- Vizualizacija arhitekture mreže
- Praćenje treninga u realnom vremenu

## ⚙️ Hiperparametri i Optimizacija

Aplikacija podržava eksperimentisanje sa:

| Parametar | Opis | Vrijednosti |
|-----------|------|-------------|
| Learning Rate | Brzina učenja | 0.00001 - 0.1 |
| Optimizer | Algoritam optimizacije | Adam, SGD, AdamW |
| Batch Size | Veličina mini-batch-a | 8 - 512 |
| Epochs | Broj epoha | 1 - 1000 |
| Dropout | Regularizacija | 0 - 0.8 |
| Hidden Layers | Arhitektura | Konfigurisano po sloju |
| Activation | Aktivaciona funkcija | ReLU, LeakyReLU, Tanh, ELU |
| Batch Norm | Batch normalizacija | Da/Ne |

### Data Augmentation
- **None** - Bez augmentacije
- **Gaussian Noise** - Dodavanje šuma
- **Oversample/SMOTE** - Balansiranje klasa
- **Both** - Kombinacija

### Outlier Handling
- **None** - Bez obrade
- **Clip (Winsorize)** - Ograničavanje na 1%/99% percentil
- **Log Transform** - Logaritamska transformacija targeta

## 📈 Evaluacija Modela

### Regresija
- **R² Score** - Koeficijent determinacije
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error

### Klasifikacija
- **Accuracy** - Tačnost
- **F1 Score (Macro)** - Harmonijska sredina precision/recall
- **Confusion Matrix** - Matrica konfuzije

## 🖥️ GPU Podrška

Aplikacija automatski detektuje CUDA uređaje i koristi GPU ako je dostupan:
- Prikaz GPU info na početnoj stranici
- Toggle za GPU/CPU u Network Lab-u

## 📸 Screenshots

### Network Lab
- Vizualizacija neuronske mreže
- Real-time praćenje treninga
- Prikaz loss kriva

### Model Comparison
- Poređenje metrika
- Side-by-side analiza
- Vizualizacije performansi

## 🔧 API Endpoints

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/health` | GET | Status servera |
| `/schema` | GET | Šema dataseta |
| `/train` | POST | Pokretanje treninga |
| `/train/{id}/status` | GET | Status treninga |
| `/train/{id}/cancel` | POST | Otkazivanje treninga |
| `/experiments` | GET | Lista eksperimenata |
| `/experiments/{id}` | GET | Detalji eksperimenta |
| `/experiments/{id}` | DELETE | Brisanje eksperimenta |
| `/experiments/{id}` | PATCH | Preimenovanje |
| `/predict` | POST | Predikcija |
| `/compare` | GET | Poređenje modela |

## 📝 Tehničke Napomene (Implementation Notes)

### Vizualizacija Mreže
- Canvas vizualizacija prikazuje **STRUKTURU** mreže (broj slojeva, neurona)
- Težine konekcija **NISU** vizualizovane (sve linije imaju istu debljinu/opacity)
- Forward-pass animacija **NIJE** implementirana - prikaz je statičan

### Feature Importance
- Koristi **perturbation-based** metodu (nije SHAP ili LIME)
- Za svaki feature: zamijeni sa baseline vrijednošću i mjeri promjenu predikcije
- Baseline vrijednosti su **median (numerički) / mode (kategorijski)** iz TRENING seta
- Procenti su normalizovani da ukupno daju 100%

### Reproducibilnost
- Random seed (default: 42) postavljen za Python, NumPy, PyTorch i CUDA
- `torch.backends.cudnn.deterministic = True` za determinističke rezultate

### Rate Limiting
- Maksimalno **2 paralelna treninga** (in-memory limit)
- Vraća HTTP 429 ako je limit dostignut

### Input Validacija
- Pydantic validacija sa range checks:
  - snaga: 30-800 HP
  - kilometraza: 0-500000 km
  - kubikaza: 500-8000 cc
  - god: 1980-2026

## 👥 Autori

Projekat razvijen za predmet "Metode dubokog učenja"

## 📄 Licenca

MIT License
