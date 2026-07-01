import re
import pandas as pd


# ======================
# 1. DICTIONNAIRES DE PRÉTRAITEMENT
# ======================

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "cant": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "dont": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'll": "it will", "it's": "it is", "let's": "let us", "might've": "might have",
    "must've": "must have", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is", "who'll": "who will",
    "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
    "wont": "will not", "doesnt": "does not", "didnt": "did not", "wasnt": "was not",
    "werent": "were not", "wouldnt": "would not", "couldnt": "could not", "shouldnt": "should not",
    "hasnt": "has not", "havent": "have not", "hadnt": "had not", "isnt": "is not",
    "arent": "are not", "im": "i am", "ive": "i have"
}

SLANG_CORRECTIONS = {
    "naw": "no", "nah": "no", "aint": "am not",
    "idk": "i do not know", "idc": "i do not care", "lol": "laughing", "lmao": "laughing",
    "tbh": "to be honest", "btw": "by the way", "omg": "oh my god",
    "ur": "your", "u": "you", "yall": "you all",
    "wanna": "want to", "gonna": "going to", "gotta": "got to", "gimme": "give me", "lemme": "let me"
}

EMOJI_MAP = {
    '😊': ' happy ', '😃': ' happy ', '😁': ' happy ', '😄': ' happy ', '😆': ' happy ',
    '🙂': ' happy ', '😍': ' love ', '❤️': ' love ', '💕': ' love ', '💖': ' love ',
    '👍': ' good ', '✅': ' yes ', '🎉': ' celebrate ', '🎊': ' celebrate ',
    '😂': ' laughing ', '🤣': ' laughing ', '💯': ' perfect ', '⭐': ' star ', '🔥': ' fire ',
    '😢': ' sad ', '😭': ' sad ', '😞': ' sad ', '😔': ' sad ', '😠': ' angry ',
    '😡': ' angry ', '🤬': ' angry ', '👎': ' bad ', '❌': ' no ', '💔': ' broken ',
    '😐': ' neutral ', '😑': ' neutral ', '🤔': ' thinking ', '❓': ' question ',
    '❗': ' exclamation '
}

# ======================
# 2. FONCTIONS DE PRÉTRAITEMENT
# ======================

def expand_contractions(text):
    for contraction, replacement in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def expand_slang(text):
    for slang, replacement in SLANG_CORRECTIONS.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text

def replace_emojis(text):
    for emoji, desc in EMOJI_MAP.items():
        text = text.replace(emoji, desc)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
                  r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF'
                  r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
                  r'\U00002702-\U000027B0\U000024C2-\U0001F251]+', ' ', text)
    return text

def handle_negations(text):
    text = re.sub(r'\b(not|no|never|none|nobody|nothing|neither|nor)\s+(\w+)', r'\1_\2', text)
    return text

def preprocess_text(text):
    """Pipeline minimaliste optimisé pour Deep Learning"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    text = expand_contractions(text)
    text = expand_slang(text)
    text = replace_emojis(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = handle_negations(text)
    text = re.sub(r'\d+', ' <NUM> ', text)
    text = re.sub(r'[^a-z\s!?\'_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ======================
# 3. PRÉTRAITEMENT DU DATASET
# ======================

def preprocess_full_dataset(filepath='sentiment_dataset.csv'):
    print("="*60)
    print("🚀 PRÉTRAITEMENT DU DATASET")
    print("="*60)
    
    # Charger le dataset
    df = pd.read_csv(filepath)
    print(f"\n✅ Dataset chargé : {len(df):,} échantillons")
    
    # Nettoyer valeurs manquantes
    df = df[df['text'].notna() & (df['text'].astype(str).str.strip() != '')].copy()
    
    # Appliquer prétraitement
    print("🧹 Prétraitement en cours...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Supprimer textes vides après prétraitement
    df = df[df['text_clean'].str.strip() != ''].copy()
    print(f"✅ {len(df):,} échantillons conservés")
    
    # Statistiques de longueur
    df['text_length'] = df['text_clean'].apply(lambda x: len(x.split()))
    max_length = int(df['text_length'].quantile(0.95))
    
    print(f"\n📏 Longueur recommandée (95e percentile) : {max_length} tokens")
    
    return df, max_length
