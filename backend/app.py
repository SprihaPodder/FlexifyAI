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


from PIL import Image, ImageStat
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
    allow_origins=["http://localhost:5173"],  
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




@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        img = Image.open(temp_path)
        width, height = img.size
        try:
            ocr_raw = pytesseract.image_to_string(img)
            ocr_text = ''.join(c for c in ocr_raw if c.isprintable())
        except Exception:
            ocr_text = "OCR failed or not available"
        
        analysis = {
            "dimensions": {"width": width, "height": height},
            "format": img.format,
            "mode": img.mode,
            "extracted_text": ocr_text.strip() if ocr_text else "No text detected"
        }
        img.close()
        os.unlink(temp_path)
        return analysis
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

@app.post("/image-features/")
async def image_features(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        img = Image.open(temp_path)

        stat = ImageStat.Stat(img)
        mode = img.mode
        features = {}
        if mode in ['RGB', 'RGBA']:
            features["mean_color"] = { 
                "R": round(stat.mean[0],2), 
                "G": round(stat.mean[1],2), 
                "B": round(stat.mean[2],2) 
            }
            features["stddev_color"] = { 
                "R": round(stat.stddev[0],2), 
                "G": round(stat.stddev[1],2), 
                "B": round(stat.stddev[2],2)
            }
        else:
            features["mean_gray"] = round(stat.mean[0],2)
            features["stddev_gray"] = round(stat.stddev[0],2)
        hist = img.histogram()
        features["histogram_preview"] = hist[:32]

        img.close()
        os.unlink(temp_path)
        return features
    except Exception as e:
        return {"error": f"Failed to extract features: {str(e)}"}


@app.post("/image-explain/")
async def image_explain(file: UploadFile = File(...)):
    try:
        explanation = {
            "explanation_steps": [
                "Image normalization and preprocessing",
                "Model extracts feature maps via convolution",
                "Model focuses on regions of highest information gain",
                "Layered feature importances are computed",
                "Final AI prediction is explained layer-by-layer (see SHAP/LIME dashboard or microservice)"
            ],
            "shap_value_stub": [
                {"area":"upper_left", "importance":0.15},
                {"area":"center", "importance":0.32},
                {"area":"lower_right", "importance":0.19}
            ],
            "note":
                "RUN true explainable AI (SHAP, LIME) in an API/microservice with isolated modern numpy and model. "
                "This stub keeps app compatible with your CSV/numpy setup."
        }
        return explanation
    except Exception as e:
        return {"error": f"Failed to provide explanation: {str(e)}"}


@app.post("/image-advanced/")
async def image_advanced(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        img = Image.open(temp_path)
        width, height = img.size

        stat = ImageStat.Stat(img)
        info = {
            "aspect_ratio": round(width / height, 3) if height else None,
            "brightness": round(np.mean(img.convert("L")), 2),
            "contrast": round(np.std(np.array(img.convert("L"))), 2),
            "entropy": round(img.convert("L").entropy(), 2) if hasattr(img, "entropy") else None
        }
        img.close()
        os.unlink(temp_path)
        return info
    except Exception as e:
        return {"error": f"Failed to perform advanced analysis: {str(e)}"}

