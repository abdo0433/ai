# ats_score.py
import re

def extract_keywords(text):
    text = text.lower()
    words = re.findall(r"[a-zA-Z]+", text)
    return list(set(words))

def ats_score(cv_text, jd_text):
    cv_keywords = extract_keywords(cv_text)
    jd_keywords = extract_keywords(jd_text)

    matched = set(cv_keywords) & set(jd_keywords)
    missing = set(jd_keywords) - matched

    score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 0
    return round(score, 2), list(matched), list(missing)