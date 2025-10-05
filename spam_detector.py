# spam_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Első futáskor letöltjük a stopword listát ---
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- 1. Adatok betöltése ---
print(" Adatok betöltése...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
##df.columns = ['label', 'message']

# --- 2. Adattisztítás ---
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

print(" Szövegek tisztítása...")
df['cleaned'] = df['message'].apply(clean_text)

# --- 3. Címkék numerikus konvertálása ---
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- 4. Tanító és teszt halmaz ---
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_num'], test_size=0.2, random_state=42
)

# --- 5. Szövegvektorok létrehozása ---
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 6. Modell tanítása ---
print(" Modell tanítása...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- 7. Előrejelzés ---
y_pred = model.predict(X_test_vec)

# --- 8. Eredmények ---
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\n Modell pontossága: {acc}%\n")
print("Klasszifikációs riport:")
print(classification_report(y_test, y_pred))

# --- 9. Konfúziós mátrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Konfúziós Mátrix')
plt.xlabel('Előrejelzett')
plt.ylabel('Valódi')
plt.colorbar()
plt.show()

# --- 10. Saját teszt ---
print("\n--- Saját üzenet tesztelése ---")
while True:
    user_msg = input("Adj meg egy üzenetet (vagy írd be: exit): ")
    if user_msg.lower() == "exit":
        print("👋 Kilépés...")
        break
    cleaned = clean_text(user_msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print("👉 Ez az üzenet:", "📩 SPAM" if pred == 1 else " NEM SPAM")
