# Medicinal Plant Identification

A computer-vision web app that identifies a plant from a leaf photograph. A **VGG19** Keras model
does the classification; a small **Flask** app serves it.

Upload a leaf image, get a predicted class back.

> ## ⚠️ This repo currently serves the *poultry* model
>
> `app.py` loads **`plD_vgg19.h5`** and maps predictions to `cocci` / `healthy` / `ncd` / `salmo` —
> which are **poultry disease** labels, not plants.
>
> The actual 78-class Ayurvedic medicinal-plant model is **`model_2_vgg19.h5`**, trained by
> [`ML_projects/Med_Plant_Detection/mePD2.py`](https://github.com/sreelekha-22/ML_projects/tree/main/Med_Plant_Detection)
> on the `FMLd/` dataset (Tulsi, Neem, Amla, Turmeric, Ashoka, Brahmi and 73 more).
>
> The `md.ipynb` notebook in this repo is the poultry training run. To make this repo match its
> name, point `app.py` at `model_2_vgg19.h5` and replace the 4-entry label map with the 78-class
> list. See [How the model is wired](#how-the-model-is-wired) below.

## How the model is wired

```
leaf image  →  resize 224×224  →  scale by 1/255  →  ImageNet preprocess_input
            →  VGG19 (plD_vgg19.h5)  →  argmax  →  class index  →  label
```

In `app.py`:

1. The uploaded file is saved with werkzeug's `secure_filename` under `uploads/`
2. Loaded and resized to `224 × 224` — the size VGG19 expects
3. Scaled by `1/255` and passed through `preprocess_input`
4. Run through the trained model; `argmax` picks the winning class
5. The class index is mapped to a label

```python
MODEL_PATH = 'plD_vgg19.h5'
model = load_model(MODEL_PATH)
```

| Class index | Label returned |
|---|---|
| 0 | `Class 1: cocci` |
| 1 | `Class 2: healthy` |
| 2 | `Class 3: ncd` |
| 3 | `Class 4: salmo` |

### Architecture the weights were trained with

Both this repo's model and the plant model use the same shape — VGG19 with the classifier head
removed and the backbone frozen, plus one dense softmax layer:

```python
vgg = VGG19(input_shape=[224, 224, 3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
```

So swapping in `model_2_vgg19.h5` is a two-line change: the `Dense` layer becomes `Dense(78)`, and
the index→label map needs the full 78-entry list.

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

`app.py` loads the weights at import time and **will fail to start without the `.h5` file**:

```python
MODEL_PATH = 'plD_vgg19.h5'
model = load_model(MODEL_PATH)
```

The `.h5` is not committed. Either place it next to `app.py`, or retrain from the notebook and save
it under that name.

### Run

```bash
mkdir -p uploads
python app.py
```

Open **http://localhost:5000** and upload a leaf image.

> Runs with `debug=True`. Bind to a proper host and disable debug before exposing it anywhere.

## Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Upload page (`templates/index.html`) |
| `POST` | `/predict` | Save the upload, run inference, return the predicted class |

## Project layout

```
├── app.py                     Flask app + VGG19 inference
├── md.ipynb                   training / evaluation run
├── Untitled0.ipynb            exploratory analysis
├── requirements.txt           pinned environment
├── templates/                 Jinja2 views (base, index)
├── static/                    css + js
└── uploads/                   uploaded images (create it; not committed)
```

## License

MIT
