import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']



# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    return ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)

# Apply preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)

# Initialize and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Oversampling minority class using resample
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

spam_upsampled = resample(spam, 
                          replace=True,     
                          n_samples=len(ham),    
                          random_state=42)  

# Combine and shuffle
upsampled = pd.concat([ham, spam_upsampled]).sample(frac=1, random_state=42)

# Create features and labels for the balanced train set
X_train_resampled = vectorizer.transform(upsampled['cleaned_message'])
y_train_resampled = upsampled['label'].map({'ham': 0, 'spam': 1}).values

# Create a balanced test set
ham_test = df[df['label'] == 'ham'].sample(n=len(spam_test := df[df['label'] == 'spam']), random_state=42)
test_set = pd.concat([ham_test, spam_test]).sample(frac=1, random_state=42)

# Transform test set
X_test_balanced = vectorizer.transform(test_set['cleaned_message'])
y_test_balanced = test_set['label'].map({'ham': 0, 'spam': 1}).values

# Initialize Random Forest model
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)  # Train on the resampled data

# Streamlit interface
st.title('Spam Classification')

# User input
user_input = st.text_area("Enter your message here:")

if st.button('Predict / Classify'):
    if user_input:
        # Preprocess and vectorize the input message
        cleaned_input = preprocess_text(user_input)
        input_vectorized = vectorizer.transform([cleaned_input])

        # Make prediction
        prediction = rf_model.predict(input_vectorized)

        # Display results
        if prediction[0] == 1:
            st.write("\U0001F6A8 This message is classified as **Spam**.")
        else:
            st.write("\U00002705 This message is classified as **Ham**.")
    else:
        st.write("Please enter a message for classification.")


#WINNER!! As a valued network customer you have been selected to receive£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.