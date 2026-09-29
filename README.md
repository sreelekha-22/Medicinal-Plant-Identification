# Medicinal Plant Identification

A computer-vision web app that identifies plant disease classes from leaf photographs. A **VGG19**
Keras model does the classification; a small **Flask** app serves it.

Upload a leaf image, get a predicted class back.

## How it works

```
leaf image  →  resize to 224×224  →  scale to [0,1]  →  ImageNet preprocess_input
            →  VGG19 (plD_vgg19.h5)  →  argmax  →  class label
```

The prediction path in `app.py`:

1. The uploaded file is saved with `werkzeug`'s `secure_filename` under `uploads/`
2. Loaded and resized to `224 × 224` — the size VGG19 expects
3. Scaled by `1/255` and passed through `preprocess_input`
4. Run through the trained model; `argmax` picks the winning class
5. The class index is mapped to a human-readable label

| Class index | Label |
|---|---|
| 0 | `Class 1: cocci` |
| 1 | `Class 2: healthy` |
| 2 | `Class 3: ncd` |
| 3 | `Class 4: salmo` |

## Tech stack

| Concern | Technology |
|---|---|
| Deep learning | TensorFlow 2.2 · Keras (VGG19) |
| Vision | OpenCV, NumPy, Pillow |
| ML | scikit-learn, pandas, matplotlib, seaborn |
| Web | Flask 1.1 · Flask-CORS |
| Notebooks | Jupyter (`md.ipynb`, `Untitled0.ipynb`) |

The notebooks hold the training and evaluation work; `app.py` is the serving layer.

## Setup

The pinned `requirements.txt` targets **Python 3.7/3.8** (TensorFlow 2.2 era). On a newer
interpreter, install TensorFlow separately and treat the pin file as a record of the original
environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### The trained model

`app.py` loads `plD_vgg19.h5` at import time and **will fail to start without it**:

```python
MODEL_PATH = 'plD_vgg19.h5'
model = load_model(MODEL_PATH)
```

The `.h5` file is not committed. Either place it next to `app.py`, or retrain from the notebook and
save it under that name.

### Run

```bash
mkdir -p uploads
python app.py
```

Open **http://localhost:5000** and upload a leaf image.

> Runs with `debug=True`. Bind to a proper host and disable debug before exposing it anywhere.

## Project layout

```
├── app.py                     Flask app + VGG19 inference
├── md.ipynb                   model training / evaluation
├── Untitled0.ipynb            exploratory analysis
├── requirements.txt           pinned environment
├── templates/                 Jinja2 views (base, index)
├── static/                    css + js
└── uploads/                   uploaded images (create it; not committed)
```

## License

MIT
