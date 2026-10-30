# Multi-Disease Medical Image Classifier

Educational prototype for classifying labelled medical images. It is not a medical diagnostic tool.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put each dataset under `data/<module>/{train,val,test}/<class>/`. Start with `chest_xray` and two classes such as `NORMAL` and `PNEUMONIA`.

To download a small balanced three-class chest X-ray subset (300 images per class):

```bash
python download_small_dataset.py
```

Train a model:

```bash
python train.py --module chest_xray --epochs 10
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Training is required for every module. The app expects the resulting `.keras` model and label `.json` file in `models/`. The original FastAPI backend is also available in `app.py` if an API is needed later.
