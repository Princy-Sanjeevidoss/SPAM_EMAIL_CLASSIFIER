# spam_classifier.py

# 📌 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📥 Load the Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])

# 🧹 Preprocess the Data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# 🔀 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🔡 Vectorize Text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# ✅ Train SVM Model
svm_model = SVC()
svm_model.fit(X_train_vec, y_train)

# 📊 Predict and Evaluate
nb_preds = nb_model.predict(X_test_vec)
svm_preds = svm_model.predict(X_test_vec)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))

print("\nNaive Bayes Classification Report:\n", classification_report(y_test, nb_preds))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds))

# 📈 Confusion Matrices
sns.heatmap(confusion_matrix(y_test, nb_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.show()

sns.heatmap(confusion_matrix(y_test, svm_preds), annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.show()

# 🧪 Test with Custom Email
sample = ["Congratulations! You've won a free iPhone. Call now!"]
sample_vec = vectorizer.transform(sample)
print("Naive Bayes Prediction:", nb_model.predict(sample_vec))

# 💾 Save the Model (optional)
import pickle
with open("spam_classifier_nb.pkl", "wb") as f:
    pickle.dump(nb_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
