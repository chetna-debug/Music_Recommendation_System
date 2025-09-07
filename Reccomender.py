import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches
from typing import List, Dict, Any, Optional

# Candidate columns; engine auto-picks what exists in your CSV
NUMERIC_CANDIDATES = [
    "acousticness","danceability","energy","instrumentalness","liveness",
    "loudness","speechiness","valence","tempo","duration_ms","popularity"
]

TEXT_CANDIDATES = [
    "track_name","name","artists","artist_name","album_name",
    "playlist_genre","playlist_subgenre","genre","genres"
]

META_CANDIDATES = [
    "track_name","name","artists","artist_name","album_name",
    "id","track_id","uri","url","preview_url","image_url","popularity"
]

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def existing_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]

class RecommendationEngine:
    def __init__(self):
        self.numeric_cols: List[str] = []
        self.text_cols: List[str] = []
        self.numeric_imputer = SimpleImputer(strategy="median")
        # with_mean=False keeps compatibility with sparse stacking
        self.numeric_scaler = StandardScaler(with_mean=False)
        self.tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, stop_words="english")
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.medians: Dict[str, float] = {}
        self.df_meta: pd.DataFrame = pd.DataFrame()
        self.fitted = False

    def _build_text_series(self, df: pd.DataFrame) -> pd.Series:
        cols = self.text_cols
        if not cols:
            return pd.Series([""] * len(df), index=df.index)
        return df[cols].astype(str).fillna("").agg(" ".join, axis=1)

    def fit(self, df: pd.DataFrame):
        # Detect columns
        self.numeric_cols = existing_cols(df, NUMERIC_CANDIDATES)
        self.text_cols = existing_cols(df, TEXT_CANDIDATES)

        if not self.numeric_cols and not self.text_cols:
            raise ValueError("No usable columns found. Provide at least some numeric audio features or text columns.")

        # Numeric block
        if self.numeric_cols:
            X_num = self.numeric_imputer.fit_transform(df[self.numeric_cols])
            # Save medians for synthetic queries
            self.medians = {c: float(m) for c, m in zip(self.numeric_cols, self.numeric_imputer.statistics_)}
            X_num = self.numeric_scaler.fit_transform(X_num)
        else:
            from scipy.sparse import csr_matrix
            X_num = csr_matrix((len(df), 0))

        # Text block
        text_series = self._build_text_series(df)
        X_txt = self.tfidf.fit_transform(text_series)

        # Stack
        X = hstack([X_num, X_txt]).tocsr()

        # Fit NN
        self.nn.fit(X)

        # Keep meta (best-effort)
        meta_cols = existing_cols(df, META_CANDIDATES)
        title_col = pick_first_existing(df, ["track_name","name"]) or ""
        artist_col = pick_first_existing(df, ["artists","artist_name"]) or ""
        meta_extra = pd.DataFrame({
            "title": df[title_col] if title_col else [""]*len(df),
            "artist": df[artist_col] if artist_col else [""]*len(df)
        })
        self.df_meta = pd.concat([df[meta_cols], meta_extra], axis=1)
        self.df_meta.reset_index(drop=True, inplace=True)

        self.fitted = True
        return X

    def _transform_rows(self, df_rows: pd.DataFrame):
        from scipy.sparse import csr_matrix
        if self.numeric_cols:
            Xn = self.numeric_imputer.transform(df_rows.reindex(columns=self.numeric_cols, fill_value=np.nan))
            Xn = self.numeric_scaler.transform(Xn)
        else:
            Xn = csr_matrix((len(df_rows), 0))

        text_series = self._build_text_series(df_rows)
        Xt = self.tfidf.transform(text_series)

        return hstack([Xn, Xt]).tocsr()

    def _make_query_row_from_mood(self, **kwargs) -> pd.DataFrame:
        data = {}
        for c in self.numeric_cols:
            data[c] = [kwargs.get(c, self.medians.get(c, 0.0))]
        for c in self.text_cols:
            data[c] = [""]
        if not data:
            data["dummy"] = [""]
        return pd.DataFrame(data)

    def _lookup_track_index(self, query: str) -> Optional[int]:
        if "title" in self.df_meta.columns:
            titles = self.df_meta["title"].astype(str)
            exact = titles[titles.str.lower() == query.lower()]
            if len(exact):
                return int(exact.index[0])
            contains = titles[titles.str.lower().str.contains(query.lower(), na=False)]
            if len(contains):
                return int(contains.index[0])
            matches = get_close_matches(query, titles.tolist(), n=1, cutoff=0.6)
            if matches:
                return int(titles[titles == matches[0]].index[0])
        return None

    def recommend_by_track(self, query_track: str, k: int = 10) -> List[Dict[str, Any]]:
        assert self.fitted, "Call fit() or load() first."
        idx = self._lookup_track_index(query_track)
        if idx is None:
            return []

        qX = self._transform_rows(self.df_meta.iloc[[idx]])
        dists, indices = self.nn.kneighbors(qX, n_neighbors=min(k+1, len(self.df_meta)))
        out = []
        for dist, i in zip(dists[0], indices[0]):
            if int(i) == int(idx):
                continue
            meta = self.df_meta.iloc[int(i)].to_dict()
            meta["score"] = float(1.0 - dist)  # cosine similarity (approx)
            out.append(meta)
            if len(out) >= k:
                break
        return out

    def recommend_by_mood(self, k: int = 10, **mood_numeric) -> List[Dict[str, Any]]:
        assert self.fitted, "Call fit() or load() first."
        
        # Debug logging
        print("\n[Mood Recommendation Request]")
        print("Received mood values:", mood_numeric)

        row = self._make_query_row_from_mood(**mood_numeric)
        qX = self._transform_rows(row)
        dists, indices = self.nn.kneighbors(qX, n_neighbors=min(k, len(self.df_meta)))
        
        out = []
        for rank, (dist, i) in enumerate(zip(dists[0], indices[0]), start=1):
            meta = self.df_meta.iloc[int(i)].to_dict()
            meta["score"] = float(1.0 - dist)
            out.append(meta)
            
            # Debug print top 5 recommendations
            if rank <= 5:
                print(f"Top {rank}: {meta.get('title')} by {meta.get('artist')} (score={meta['score']:.4f})")

        return out

    def save(self, path_joblib: str, feature_meta_path: str):
        payload = {
            "numeric_cols": self.numeric_cols,
            "text_cols": self.text_cols,
            "medians": self.medians,
            "numeric_imputer": self.numeric_imputer,
            "numeric_scaler": self.numeric_scaler,
            "tfidf": self.tfidf,
            "nn": self.nn,
            "df_meta": self.df_meta
        }
        joblib.dump(payload, path_joblib)
        with open(feature_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "numeric_cols": self.numeric_cols,
                "text_cols": self.text_cols
            }, f, indent=2)

    def load(self, path_joblib: str):
        payload = joblib.load(path_joblib)
        self.numeric_cols = payload["numeric_cols"]
        self.text_cols = payload["text_cols"]
        self.medians = payload["medians"]
        self.numeric_imputer = payload["numeric_imputer"]
        self.numeric_scaler = payload["numeric_scaler"]
        self.tfidf = payload["tfidf"]
        self.nn = payload["nn"]
        self.df_meta = payload["df_meta"]
        self.fitted = True
        return self
