from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import Optional

import json

import numpy as np
np.random.seed(42)

import pandas as pd

from fastapi.responses import JSONResponse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


from PIL import Image, ImageFilter, ImageStat
import pytesseract
import chardet
from transformers import pipeline


import textstat
import spacy
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk

from collections import Counter
from typing import Dict, List, Any


import spacy
import textstat
import nltk
from collections import Counter
from typing import Dict, List, Any


import re
import unicodedata
from docx import Document




from io import BytesIO
from sklearn.cluster import KMeans




MODEL_MAP = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "gradient_boosting": GradientBoostingClassifier,
    "knn": KNeighborsClassifier,
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-url.onrender.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')




@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Test endpoint working"}


def extract_docx_text(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    doc = Document(temp_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    os.unlink(temp_path)
    return text

def clean_text_for_display(text):
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'[^\x20-\x7E\n\t]', '', text)


def fill_missing_safely(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric with mean, non-numeric with mode (if present)."""
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    cat_cols = df.columns.difference(num_cols)
    for c in cat_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            if len(mode_vals) > 0:
                df[c] = df[c].fillna(mode_vals.iloc[0])
            else:
                df[c] = df[c].fillna("missing")
    return df

def zscore_mask(df_numeric: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask for rows that are NOT outliers by Z-score across numeric columns."""
    if df_numeric.shape[1] == 0:
        return pd.Series(True, index=df_numeric.index)
    mask = pd.Series(True, index=df_numeric.index)
    for col in df_numeric.columns:
        s = df_numeric[col]
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        z = (s - s.mean()) / std
        col_mask = z.abs() < threshold
        col_mask = col_mask.fillna(True)
        mask &= col_mask
    return mask

def iqr_mask(df_numeric: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """Return boolean mask for rows that are NOT outliers by IQR across numeric columns."""
    if df_numeric.shape[1] == 0:
        return pd.Series(True, index=df_numeric.index)
    mask = pd.Series(True, index=df_numeric.index)
    for col in df_numeric.columns:
        s = df_numeric[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        col_mask = (s >= lower) & (s <= upper)
        col_mask = col_mask.fillna(True)
        mask &= col_mask
    return mask



@app.post("/upload/")
async def upload_dataset(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_column: str = Form(...),
    model_params: Optional[str] = Form(None),      
    handle_missing: str = Form("drop"),            
    handle_outliers: str = Form("none"),           
):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    file_size_kb = os.path.getsize(temp_path) / 1024.0

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in CSV."}

    rows_before, cols = df.shape

    
    if handle_missing == "drop":
        df = df.dropna()
    elif handle_missing == "mean":
        df = fill_missing_safely(df)
    elif handle_missing == "none":
        pass
    else:
        return {"error": f"Invalid handle_missing option: {handle_missing}. Use 'none' | 'drop' | 'mean'."}

    
    if target_column not in df.columns or df.empty:
        return {"error": "After cleaning, dataset is empty or target column missing."}

    X_tmp = df.drop(columns=[target_column], errors="ignore")
    num_feats = X_tmp.select_dtypes(include="number")

    mask = pd.Series(True, index=df.index)
    if handle_outliers == "zscore":
        mask = zscore_mask(num_feats)
    elif handle_outliers == "iqr":
        mask = iqr_mask(num_feats)
    elif handle_outliers == "none":
        pass
    else:
        return {"error": f"Invalid handle_outliers option: {handle_outliers}. Use 'none' | 'zscore' | 'iqr'."}

    df = df[mask]

    
    rows_after = len(df)
    rows_removed = rows_before - rows_after

    if rows_after < 2:
        return {"error": "Not enough rows after cleaning to train a model (need at least 2)."}

    
    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]

   
    if model_params:
        try:
            params = json.loads(model_params)
            
            for k, v in params.items():
                if isinstance(v, str) and v.isdigit():
                    params[k] = int(v)
                else:
                    try:
                        float_val = float(v)
                        if '.' in v:
                            params[k] = float_val
                        else:
                            params[k] = int(float_val)
                    except Exception:
                        pass  
        except Exception as e:
            return {"error": f"Failed to parse model_params: {e}"}
    else:
        params = {}

    
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', KNNImputer() if params.get('imputation', '') == 'knn' else SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )


    valid_params = {
        'decision_tree': ['max_depth', 'min_samples_split'],
        'random_forest': ['n_estimators', 'max_depth', 'min_samples_split'],
        'logistic_regression': ['C', 'max_iter'],
        'svm': ['C', 'kernel'],
        'gradient_boosting': ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split'],
        'knn': ['n_neighbors', 'weights']
    }

    ModelClass = MODEL_MAP.get(model_type)
    if ModelClass is None:
        return {"error": f"Invalid model type '{model_type}'."}

    
    filtered_params = {k: v for k, v in params.items() if k in valid_params.get(model_type, [])}

    try:
        model = ModelClass(**filtered_params)
    except Exception as e:
        return {"error": f"Error initializing model: {e}"}

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])


    cv = min(5, rows_after)
    if cv < 2:
        return {"error": "Not enough samples to perform cross-validation."}

   
    if params.get('auto_tune', False):
        
        param_distributions = {}
        if model_type in ['random_forest', 'gradient_boosting']:
            param_distributions = {
                'classifier__n_estimators': randint(50, 300),
                'classifier__max_depth': randint(3, 16),
                'classifier__min_samples_split': randint(2, 20),
            }
        elif model_type == 'knn':
            param_distributions = {
                'classifier__n_neighbors': randint(3, 30),
                'classifier__weights': ['uniform', 'distance']
            }
        elif model_type == 'svm':
            param_distributions = {
                'classifier__C': uniform(0.1, 10),
                'classifier__kernel': ['linear', 'rbf', 'poly']
            }
     

        if param_distributions:
            search = RandomizedSearchCV(pipe, param_distributions, n_iter=25, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
            search.fit(X, y)
            best_model = search.best_estimator_
            scores = cross_val_score(best_model, X, y, cv=cv)
        else:
       
            scores = cross_val_score(pipe, X, y, cv=cv)
    else:
        scores = cross_val_score(pipe, X, y, cv=cv)

    return {
        "accuracy": round(scores.mean() * 100, 2),
        "std_dev": round(scores.std() * 100, 2),
        "model_used": model_type,
        "params_used": params,
        "filename": file.filename,
        "file_size_kb": round(file_size_kb, 2),
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "rows_removed": int(rows_removed),
        "columns": int(cols),
        "handle_missing": handle_missing,
        "handle_outliers": handle_outliers,
    }


@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = "numerical"
            else:
                col_type = "categorical"
            cols.append({"name": col, "type": col_type})
        return {"columns": cols}
    except Exception as e:
        return {"error": f"Failed to analyze CSV: {str(e)}"}


@app.post("/eda/")
async def eda(
    file: UploadFile = File(...),
    columns: str = Form(...),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}

    try:
        columns_info = json.loads(columns)
    except Exception as e:
        return {"error": f"Failed to parse columns: {e}"}

    if not columns_info or not isinstance(columns_info, list):
        return {"error": "Invalid columns info."}

    col_names = [c["name"] for c in columns_info]
    col_types = [c["type"] for c in columns_info]
    if not all(name in df.columns for name in col_names):
        return {"error": "Some columns not found in dataset."}
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    plt.clf()
    img_b64 = None

    try:
        if len(col_names) == 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            t1, t2 = col_types
            c1, c2 = col_names
            if t1 == "numerical" and t2 == "numerical":
                sns.scatterplot(data=df, x=c1, y=c2, ax=ax)
                ax.set_title(f"Scatterplot: {c1} vs {c2}")
            elif t1 == "categorical" and t2 == "numerical":
                sns.boxplot(data=df, x=c1, y=c2, ax=ax)
                ax.set_title(f"Boxplot: {c2} by {c1}")
            elif t1 == "numerical" and t2 == "categorical":
                sns.boxplot(data=df, x=c2, y=c1, ax=ax)
                ax.set_title(f"Boxplot: {c1} by {c2}")
            elif t1 == "categorical" and t2 == "categorical":
                ct = pd.crosstab(df[c1], df[c2])
                ct.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(f"Stacked Bar: {c1} vs {c2}")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
        elif len(col_names) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            t1 = col_types[0]
            c1 = col_names[0]
            if t1 == "numerical":
                sns.histplot(df[c1], kde=True, ax=ax)
                ax.set_title(f"Histogram: {c1}")
            else:
                df[c1].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Barplot: {c1}")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
        elif len(col_names) >= 3:
            num_cols = [c["name"] for c in columns_info if c["type"] == "numerical"]
            cat_cols = [c["name"] for c in columns_info if c["type"] == "categorical"]

            if len(num_cols) == len(col_names):
                sns_plot = sns.pairplot(df[num_cols])
                buf = BytesIO()
                sns_plot.fig.suptitle("Pairplot of Numerical Columns", y=1.02)
                plt.tight_layout()
                sns_plot.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close('all')
            elif len(cat_cols) == len(col_names):
                ct = pd.crosstab([df[c] for c in cat_cols[:-1]], df[cat_cols[-1]])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Heatmap: {'/'.join(cat_cols[:-1])} vs {cat_cols[-1]}")
                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
            else:
                if len(num_cols) >= 2 and len(cat_cols) >= 1:
                    plot_cols = num_cols + [cat_cols[0]]
                    sns_plot = sns.pairplot(df[plot_cols], hue=cat_cols[0])
                    buf = BytesIO()
                    sns_plot.fig.suptitle(f"Pairplot (hue={cat_cols[0]})", y=1.02)
                    plt.tight_layout()
                    sns_plot.savefig(buf, format="png")
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
                    plt.close('all')
                else:
                    return {"error": "For 3+ columns, need at least 2 numerical and 1 categorical for mixed pairplot."}
        else:
            return {"error": "Please select at least 1 column for EDA."}
    except Exception as e:
        return {"error": f"Failed to generate plot: {e}"}

    return JSONResponse(content={
        "image": img_b64,
        "totalRows": total_rows,
        "totalColumns": total_columns,
        "missingValues": missing_values,
        "duplicateRows": duplicate_rows
    })

@app.post("/process-text/")
async def process_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".docx":
            text = extract_docx_text(content)
            encoding = 'docx'
        else:
            detection = chardet.detect(content)
            encoding = detection['encoding'] or 'utf-8'
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text = content.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return {"error": "Could not decode file with any supported encoding"}
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        word_count = len(text.split())
        line_count = len(text.splitlines())
        char_count = len(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
            sentiment = sentiment_analyzer(text[:512])[0]
        except Exception as e:
            sentiment = {"label": "N/A", "score": 0}

        return {
            "statistics": {
                "words": word_count,
                "lines": line_count,
                "characters": char_count,
                "sentences": sentence_count,
                "detected_encoding": encoding
            },
            "sentiment": sentiment,
            "preview": text[:200] + "..." if len(text) > 200 else text
        }
    except Exception as e:
        return {"error": f"Failed to process text: {str(e)}"}

@app.post("/analyze-readability/")
async def analyze_readability(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        
        analysis = {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
            "smog_index": float(textstat.smog_index(text)),
            "automated_readability_index": float(textstat.automated_readability_index(text)),
            "coleman_liau_index": float(textstat.coleman_liau_index(text)),
            "dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),
            "difficult_words": int(textstat.difficult_words(text)),
            "linsear_write_formula": float(textstat.linsear_write_formula(text)),
            "text_standard": str(textstat.text_standard(text))
        }
        
        return analysis
    except Exception as e:
        print(f"Readability analysis error: {str(e)}")  # Add logging
        return {"error": f"Failed to analyze readability: {str(e)}"}

@app.post("/extract-topics/")
async def extract_topics(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        doc = nlp(text)
        noun_phrases = [
            chunk.text for chunk in doc.noun_chunks 
            if len(chunk.text.split()) > 1
        ]
        keywords = [
            token.text.lower() for token in doc 
            if not token.is_stop and token.is_alpha and len(token.text) > 2
        ]
        word_freq = Counter(keywords)
        common_words = [
            [word, count] for word, count in word_freq.most_common(15)
        ]
        main_topics = list(set([
            ent.text for ent in doc.ents 
            if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "TOPIC"]
            and len(ent.text) > 2
        ]))
        
        return {
            "noun_phrases": list(set(noun_phrases))[:15],
            "common_words": common_words,
            "main_topics": main_topics[:10]
        }
    except Exception as e:
        print(f"Topic extraction error: {str(e)}")
        return {"error": f"Failed to extract topics: {str(e)}"}

@app.post("/extract-entities/")
async def extract_entities(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        doc = nlp(text)
        entity_categories = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),
            "DATE": set(),
            "TIME": set(),
            "MONEY": set(),
            "PERCENT": set()
        }
        
      
        for ent in doc.ents:
            if ent.label_ in entity_categories:

                clean_ent = ent.text.strip()
                if clean_ent and len(clean_ent) > 1:
                    entity_categories[ent.label_].add(clean_ent)
        return {
            category: list(entities)[:10]
            for category, entities in entity_categories.items()
        }
    except Exception as e:
        print(f"Entity extraction error: {str(e)}") 
        return {"error": f"Failed to extract entities: {str(e)}"}


def _img_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _plot_histogram_as_base64(hist_vals, title="Color Histogram"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(range(len(hist_vals)), hist_vals)
    ax.set_title(title)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def _rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))


# ---------------- Existing endpoints (process-image, extract-image-features, explain-image) ----------------
@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path)
        width, height = img.size
        fmt = img.format or "PNG"
        mode = img.mode

        extracted_text = ""
        try:
            ocr_img = img.convert("RGB")
            extracted_text = pytesseract.image_to_string(ocr_img).strip()
        except Exception:
            extracted_text = ""

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "dimensions": {"width": width, "height": height},
            "format": fmt,
            "mode": mode,
            "extracted_text": extracted_text or ""
        }

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

@app.post("/extract-image-features/")
async def extract_image_features(file: UploadFile = File(...), k_colors: int = Form(3)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")
        width, height = img.size

        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        arr = np.array(img)
        pixels = arr.reshape(-1, 3).astype(np.float32)
        max_samples = 30000
        sample_pixels = pixels[np.random.choice(pixels.shape[0], min(pixels.shape[0], max_samples), replace=False)]

        try:
            km = KMeans(n_clusters=max(1, int(k_colors)), random_state=42)
            km.fit(sample_pixels)
            centers = km.cluster_centers_.astype(int)
            dominant_hex = [_rgb_to_hex(tuple(center)) for center in centers]
        except Exception:
            avg_col = tuple(np.mean(sample_pixels, axis=0).astype(int).tolist())
            dominant_hex = [_rgb_to_hex(avg_col)]

        hist_r, _ = np.histogram(arr[:,:,0].flatten(), bins=32, range=(0,255))
        hist_g, _ = np.histogram(arr[:,:,1].flatten(), bins=32, range=(0,255))
        hist_b, _ = np.histogram(arr[:,:,2].flatten(), bins=32, range=(0,255))
        hist_sum = (hist_r + hist_g + hist_b).tolist()
        histogram_base64 = _plot_histogram_as_base64(hist_sum, title="Color Intensity Histogram")

        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_b64 = _img_to_base64(edges, fmt="PNG")

        palette_img = Image.new("RGB", (len(dominant_hex)*40, 40))
        for i, hx in enumerate(dominant_hex):
            color = tuple(int(hx.lstrip("#")[j:j+2], 16) for j in (0,2,4))
            sw = Image.new("RGB", (40,40), color)
            palette_img.paste(sw, (i*40, 0))
        palette_b64 = _img_to_base64(palette_img, fmt="PNG")

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "dominant_colors": dominant_hex,
            "palette_image": palette_b64,
            "histogram_image": histogram_base64,
            "avg_brightness": round(avg_brightness, 2),
            "contrast": round(contrast, 2),
            "edge_image": edge_b64,
            "width": width,
            "height": height,
            "format": img.format or "PNG"
        }
    except Exception as e:
        return {"error": f"Failed to extract features: {str(e)}"}

@app.post("/explain-image/")
async def explain_image(file: UploadFile = File(...), explanation_level: str = Form("short")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")

        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges).astype(np.float32)
        edges_norm = (edges_arr / edges_arr.max()) * 255.0 if edges_arr.max() > 0 else edges_arr
        edges_vis = np.stack([edges_norm, edges_norm, edges_norm], axis=-1).astype(np.uint8)
        saliency_img = Image.fromarray(edges_vis)
        saliency_b64 = _img_to_base64(saliency_img, fmt="PNG")

        steps = [
            "1) Input image is loaded and converted to RGB.",
            "2) Basic pixel-level statistics are computed (brightness / contrast).",
            "3) Edge detection (simple FIND_EDGES) highlights high-frequency regions (text, boundaries).",
            "4) Color clustering (KMeans) identifies dominant palette colors.",
            "5) OCR (if requested previously) extracts embedded text via Tesseract.",
            "6) Results are packaged into human-readable features and visual maps."
        ]
        if explanation_level == "short":
            steps = steps[:3]

        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])
        feature_summary = {
            "avg_brightness": round(avg_brightness, 2),
            "contrast": round(contrast, 2),
            "size": {"width": img.width, "height": img.height},
            "mode": img.mode,
        }

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "explanation_steps": steps,
            "saliency_image": saliency_b64,
            "feature_summary": feature_summary
        }

    except Exception as e:
        return {"error": f"Failed to generate explanation: {str(e)}"}

# ---------------- New advanced-analysis + specialized endpoints ----------------

def _detect_domains_from_features(features: dict, ocr_text: str):
    """
    Heuristic domain detection:
    - If OCR text length > threshold -> 'document'
    - If greens dominate -> 'nature'
    - If high contrast and compact palette -> 'product'
    - If portrait ratio -> 'portrait'
    - fallback -> 'general'
    """
    domains = []
    txt_len = len(ocr_text.strip()) if ocr_text else 0

    if txt_len > 20:
        domains.append("document")

    dom_colors = [c.lstrip("#") for c in features.get("dominant_colors", [])]
    hex_vals = []
    for h in dom_colors:
        try:
            r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            hex_vals.append((r,g,b))
        except Exception:
            continue

    greens = sum(1 for (r,g,b) in hex_vals if g > r and g > b and g > 100)
    if greens >= 1 and "nature" not in domains:
        domains.append("nature")

    if features.get("contrast", 0) > 30 and len(hex_vals) <= 4 and "product" not in domains:
        domains.append("product")

    size = features.get("size") or {"width": features.get("width"), "height": features.get("height")}
    try:
        w = int(size.get("width") or 0); h = int(size.get("height") or 0)
        if h > 1.2 * w and "portrait" not in domains:
            domains.append("portrait")
    except Exception:
        pass

    ordered = []
    for d in domains:
        if d not in ordered:
            ordered.append(d)
    if not ordered:
        ordered = ["general"]
    return ordered

@app.post("/advanced-analysis/")
async def advanced_analysis(file: UploadFile = File(...), k_colors: int = Form(4)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")
        width, height = img.size
        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        arr = np.array(img)
        pixels = arr.reshape(-1,3).astype(np.float32)
        max_samples = 30000
        sample_pixels = pixels[np.random.choice(pixels.shape[0], min(pixels.shape[0], max_samples), replace=False)]

        try:
            km = KMeans(n_clusters=max(1, int(k_colors)), random_state=42)
            km.fit(sample_pixels)
            centers = km.cluster_centers_.astype(int)
            dominant_hex = ['#%02x%02x%02x' % tuple(c) for c in centers]
        except Exception:
            avg_col = tuple(np.mean(sample_pixels, axis=0).astype(int).tolist())
            dominant_hex = ['#%02x%02x%02x' % tuple(avg_col)]

        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(img.convert("RGB")).strip()
        except Exception:
            ocr_text = ""

        feature_summary = {
            "avg_brightness": round(avg_brightness,2),
            "contrast": round(contrast,2),
            "size": {"width": width, "height": height},
            "mode": img.mode
        }

        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges).astype(np.float32)
        edges_norm = (edges_arr / edges_arr.max()) * 255.0 if edges_arr.max()>0 else edges_arr
        saliency_img = Image.fromarray(np.stack([edges_norm, edges_norm, edges_norm], axis=-1).astype(np.uint8))
        saliency_b64 = _img_to_base64(saliency_img)

        feat_obj = {
            "dominant_colors": dominant_hex,
            "avg_brightness": round(avg_brightness,2),
            "contrast": round(contrast,2),
            "width": width,
            "height": height,
            "size": {"width": width, "height": height}
        }

        domains = _detect_domains_from_features(feat_obj, ocr_text)

        action_map = {
            "document": [
                {"id":"table_extract", "label":"Table / Layout Extraction", "description":"Try to extract tables, key-value fields and layout from document-like images.", "endpoint":"/specialized/table-extract/"},
                {"id":"document_ner", "label":"Document Entity Extraction", "description":"Run NER and structured parsing tailored to invoices/receipts.", "endpoint":"/specialized/document-ner/"}
            ],
            "product": [
                {"id":"logo_detect", "label":"Logo / Branding Detection", "description":"Detect logos, brand names and packaging text.", "endpoint":"/specialized/logo-detect/"},
                {"id":"color_palette", "label":"Packaging Color Analysis", "description":"Detailed color breakdown for packaging design.", "endpoint":"/specialized/packaging-analysis/"}
            ],
            "nature": [
                {"id":"plant_id", "label":"Plant/Species Detector (placeholder)", "description":"Attempt to identify plant/flower types using color/shape heuristics or models.", "endpoint":"/specialized/plant-id/"},
            ],
            "portrait": [
                {"id":"face_attributes", "label":"Face Attributes (age/gender)", "description":"Estimate facial attributes (placeholder).", "endpoint":"/specialized/face-attributes/"},
            ],
            "general": [
                {"id":"edge_stats", "label":"Edge / Texture Statistics", "description":"Deeper texture analysis and image quality metrics.", "endpoint":"/specialized/texture-analysis/"}
            ]
        }

        suggested = []
        seen = set()
        for d in domains:
            for a in action_map.get(d, action_map.get("general", [])):
                if a["id"] not in seen:
                    suggested.append(a)
                    seen.add(a["id"])

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "features": feat_obj,
            "explanation": {
                "explanation_steps": [
                    "Loaded image and computed brightness/contrast.",
                    "Computed dominant colors via clustering.",
                    "Computed saliency via simple edge detection."
                ],
                "saliency_image": saliency_b64,
                "feature_summary": feature_summary
            },
            "ocr_text": ocr_text,
            "domain_suggestions": domains,
            "suggested_actions": suggested
        }

    except Exception as e:
        return {"error": f"advanced analysis failed: {str(e)}"}

# ---------------- Placeholder specialized endpoints ----------------

@app.post("/specialized/table-extract/")
async def specialized_table_extract(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path)
        text = ""
        try:
            text = pytesseract.image_to_string(img).strip()
        except Exception:
            text = ""

        lines = [l for l in text.splitlines() if l.strip()]
        rows = []
        for ln in lines:
            if any(ch.isdigit() for ch in ln) or ('|' in ln) or (',' in ln and any(c.isdigit() for c in ln)):
                rows.append(ln.strip())

        try:
            os.unlink(tmp_path)
        except:
            pass

        return {"rows": rows, "raw_text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/logo-detect/")
async def specialized_logo_detect(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        text = ""
        try:
            text = pytesseract.image_to_string(img).strip()
        except:
            text = ""
        arr = np.array(img)
        hist_r, _ = np.histogram(arr[:,:,0].flatten(), bins=8, range=(0,255))
        hist_g, _ = np.histogram(arr[:,:,1].flatten(), bins=8, range=(0,255))
        hist_b, _ = np.histogram(arr[:,:,2].flatten(), bins=8, range=(0,255))
        colors = {
            "r_hist": hist_r.tolist(),
            "g_hist": hist_g.tolist(),
            "b_hist": hist_b.tolist()
        }
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"ocr_text": text, "color_hist": colors}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/plant-id/")
async def specialized_plant_id(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        arr = np.array(img)
        avg = tuple(np.mean(arr.reshape(-1,3), axis=0).astype(int).tolist())
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"message": "Plant ID placeholder — result may be inaccurate", "avg_color": '#%02x%02x%02x' % tuple(avg)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/face-attributes/")
async def specialized_face_attributes(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path)
        w,h = img.size
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"note":"Face attributes placeholder — use a face detector model for production", "width": w, "height": h}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/texture-analysis/")
async def specialized_texture_analysis(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("L")
        arr = np.array(img).astype(np.float32)
        # simple texture metrics: mean, std, entropy-like approx
        mean = float(arr.mean())
        std = float(arr.std())
        # approximate entropy by histogram
        hist, _ = np.histogram(arr.flatten(), bins=64, range=(0,255))
        probs = hist / (hist.sum() + 1e-9)
        entropy = -float((probs * np.log(probs + 1e-9)).sum())
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"mean": round(mean,2), "std": round(std,2), "entropy": round(entropy,3)}
    except Exception as e:
        return {"error": str(e)}
