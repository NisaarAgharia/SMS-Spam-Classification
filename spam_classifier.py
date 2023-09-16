import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

print("Loading and preprocessing data...")
data = pd.read_csv('C:\\Users\\myfir\\Desktop\\spam_classifer\\spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

data['processed_message'] = data['message'].apply(preprocess_text)

print("Vectorizing data...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['processed_message'])
y = data['label'].map({'ham': 0, 'spam': 1})

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

print("Predicting on the test set...")
y_pred = nb_classifier.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

def classify_message(message):
    processed = preprocess_text(message)
    vectorized = tfidf_vectorizer.transform([processed])
    prediction = nb_classifier.predict(vectorized)
    return "Spam" if prediction[0] else "Ham"

# Example of how to classify a new message
example_messages = [
    'Congrats! You have won 1 crore',
    'Hey, how are you doing?',
    'Urgent! Call us now for a special offer',
    'I left my keys at your place yesterday',
    'Win a free iPhone now! Click this link'
]

for msg in example_messages:
    print(f'"{msg}" is classified as {classify_message(msg)}')

