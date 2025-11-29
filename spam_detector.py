import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Első futáskor lekell tölteni a stopword listát
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- 1. Spam Adatok betöltése 
print("Adatok betöltése...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# --- 2. Adattisztítás
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    try:
        sw = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        sw = set(stopwords.words('english'))
    words = [w for w in words if w not in sw]
    return ' '.join(words)

print(" Szövegek tisztítása...")
df['cleaned'] = df['message'].apply(clean_text)

#Címkék numerikus konvertálása
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

#Tanító és teszt halmaz
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_num'], test_size=0.2, random_state=42
)

#Szövegvektorok létrehozása
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Modell tanítása
print(" Modell tanítása...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

#Előrejelzés
y_pred = model.predict(X_test_vec)

#Eredmények 
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\n Modell pontossága: {acc}%\n")
print(" Klasszifikációs riport:")
print(classification_report(y_test, y_pred))

#Konfúziós mátrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Konfúziós Mátrix')
plt.xlabel('Előrejelzett')
plt.ylabel('Valódi')
plt.colorbar()
plt.show()

#Emaillel üzenettel tesztelés
print("\n--- Saját e-mail tesztelése ---")
print("Használat:")
print(" - Fájl beolvasása: file:<fájlnév> (pl. file:email1.txt)")
print(" - Többsoros beillesztés / gépelés: majd egy új sorban kell egy '.' majd nyomj ENTER-t")
print(" - Kilépés: exit\n")

spam_folder = "spam_emails"
if not os.path.exists(spam_folder):
    os.makedirs(spam_folder)

while True:
    user_input = input(" Adj meg 'file:<fájlnév>' vagy kezd el beírni/illesszed a levelet (vagy 'exit'): ").strip()

    if user_input.lower() == "exit":
        print(" Kilépés...")
        break

    #fájl esetén
    if user_input.lower().startswith("file:"):
        filepath = user_input.split("file:", 1)[1].strip()
        if not os.path.exists(filepath):
            print(" A megadott fájl nem található:", filepath)
            continue
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            email_text = f.read()
        print(f" Beolvasva: {filepath}")

    #üres sor esetén, többsoros inputként kell megadni
    elif user_input == "":
        print("írd be vagy illeszd az e-mailt. Majd egy külön sorban '.' szükséges majd ENTER.")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        email_text = "\n".join(lines)

    #Több sor esetén kell egy lezáró '.'
    else:
        print("Ha szeretnéd folytatni a levelet, folytasd a gépelést; ha vége: egy külön sorban '.' szükséges majd ENTER.")
        lines = [user_input]
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        email_text = "\n".join(lines)

    #ha nincs tartalma az emailnek akkor újra próbálja
    if not email_text.strip():
        print("Üres e-mail — próbáld újra.")
        continue

    #szöveg előfeldolgozás 
    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    if pred == 1:
        print("Ez az e-mail:  SPAM")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = f"spam_{timestamp}.txt"
        filepath = os.path.join(spam_folder, safe_name)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Detected: {datetime.now().isoformat()}\n\n")
            f.write("Original email:\n")
            f.write(email_text + "\n\n")
            f.write("Cleaned text:\n")
            f.write(cleaned + "\n")
        print(f" A spam e-mail elmentve ide: {filepath}")
    else:
        print("Ez az e-mail: NEM SPAM")
