# Medicinal Plant Identification

A computer-vision web app that identifies an **Ayurvedic medicinal plant** from a leaf photograph. A
**VGG19** Keras model trained on **80 plant species** does the classification; a small **Flask** app
serves it.

Upload a leaf image, get back the plant name and a confidence score.

## What it classifies

The model recognises **80 species** — the Ayurvedic medicinals and common edible plants:

Tulsi · Neem · Amla · Turmeric · Ashoka · Brahmi · Drumstick · Curry · Mango · Guava · Papaya ·
Rose · Jasmine · Henna · Hibiscus · Lemon · Lemongrass · Mint · Marigold · Coconut-family and
medicinal species, and more. The full ordered list lives in `CLASS_NAMES` in `app.py`.

The four `cocci` / `healthy` / `ncd` / `salmo` labels this repo used to return were **poultry
disease** classes — see [History](#history) below.

## How the model is wired

```
leaf image  →  resize 224×224  →  scale by 1/255  →  ImageNet preprocess_input
            →  VGG19 (model_2_vgg19.h5)  →  argmax  →  class index  →  plant name
```

In `app.py`:

1. The uploaded file is saved with werkzeug's `secure_filename` under `uploads/`
2. Loaded and resized to `224 × 224` — the size VGG19 expects
3. Scaled by `1/255` and passed through `preprocess_input`
4. Run through the trained model; `argmax` picks the winning class
5. The index is looked up in `CLASS_NAMES`, and the top probability is returned as a confidence %

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_2_vgg19.h5')

idx = int(np.argmax(preds[0]))
return {
    'class_index': idx,
    'class_name': CLASS_NAMES[idx],
    'confidence': round(float(preds[0][idx]) * 100.0, 2),
}
```

The model path resolves relative to `app.py`, so the app runs from any working directory. On startup
the model output width is checked against `len(CLASS_NAMES)` and the app refuses to start on a
mismatch — so a wrong-weights model fails loudly instead of returning nonsense labels.

### Training architecture

VGG19 with the classifier head removed and the backbone frozen, plus one dense softmax layer:

```python
vgg = VGG19(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
```

`len(folders)` comes from `glob('FMLd/*')` — one folder per species, 80 of them. Trained with
categorical crossentropy + Adam, `ImageDataGenerator` augmentation (rescale, shear 0.2, zoom 0.2,
horizontal flip), an 80/20 split and `EarlyStopping` on `val_loss`.

Training script: [`ML_projects/Med_Plant_Detection/mePD2.py`](https://github.com/sreelekha-22/ML_projects/tree/main/Med_Plant_Detection)

## API

| Method | Route | Response |
|---|---|---|
| `GET` | `/` | Upload page (`templates/index.html`) |
| `POST` | `/predict` | `{ "class_index": 78, "class_name": "Turmeric", "confidence": 91.4 }` |

A missing or empty file returns `400` with an explanatory message rather than a traceback.

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

`app.py` loads the weights at import time and **will fail to start without `model_2_vgg19.h5`**
present next to it. The `.h5` is not committed, so either drop the file in, or retrain it:

```bash
# in the training project, with the FMLd dataset in place
python mePD2.py          # writes model_2_vgg19.h5
# then copy it here
```

### Run

```bash
python app.py            # creates uploads/ on first upload
```

Open **http://localhost:5000** and upload a leaf image.

> Runs with `debug=True`. Bind to a proper host and disable debug before exposing it anywhere.

## Project layout

```
├── app.py                     Flask app + VGG19 inference + CLASS_NAMES
├── md.ipynb                   training / evaluation run
├── Untitled0.ipynb            exploratory analysis
├── requirements.txt           pinned environment
├── templates/                 Jinja2 views (base, index)
├── static/                    css + js
├── uploads/                   uploaded images (auto-created, not committed)
└── model_2_vgg19.h5           trained weights (not committed)
```

## History

This app previously loaded **`plD_vgg19.h5`** and returned four poultry-disease labels (`cocci`,
`healthy`, `ncd`, `salmo`) — weights trained on the `plD` dataset in
[`ML_projects/Poultry_disease_detection`](https://github.com/sreelekha-22/ML_projects/tree/main/Poultry_disease_detection).
That did not match what the repo name promises, so the app now loads the 80-species
`model_2_vgg19.h5` instead. The poultry classifier remains available in the ML_projects repo.

## License

MIT
