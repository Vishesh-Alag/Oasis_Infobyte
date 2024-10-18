import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Check for missing values and distribution of labels
print('\nChecking Missing Values : ')
print(df.isnull().sum())
print('\nChecking Distribution of Labels (Spam and Ham) : ')
print(df['label'].value_counts())

# Visualize the distribution of spam and ham messages
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam and Ham Messages')
plt.show()

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
#df.to_csv('cleaned_msg_spam.csv', index=False)

# Initialize and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Oversampling minority class using resample
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

spam_upsampled = resample(spam, 
                          replace=True,     
                          n_samples=len(ham),    
                          random_state=42)  

# Combine and shuffle
upsampled = pd.concat([ham, spam_upsampled]).sample(frac=1, random_state=42)
# Visualize the distribution of spam and ham messages after oversampling
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='label', data=upsampled)
plt.title('Distribution of Spam and Ham Messages After Oversampling')
plt.xlabel('Label')
plt.ylabel('Count')

# Add counts on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.show()
# Create features and labels for the balanced train set
X_train_resampled = vectorizer.transform(upsampled['cleaned_message'])
y_train_resampled = upsampled['label'].map({'ham': 0, 'spam': 1}).values

# Create a balanced test set
ham_test = df[df['label'] == 'ham'].sample(n=len(spam_test := df[df['label'] == 'spam']), random_state=42)
test_set = pd.concat([ham_test, spam_test]).sample(frac=1, random_state=42)

# Transform test set
X_test_balanced = vectorizer.transform(test_set['cleaned_message'])
y_test_balanced = test_set['label'].map({'ham': 0, 'spam': 1}).values

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)  # Train on the resampled data
    y_pred_balanced = model.predict(X_test_balanced)  # Make predictions

    # Evaluate the model
    print(f"\n{model_name} Evaluation:")
    print(confusion_matrix(y_test_balanced, y_pred_balanced))
    print(classification_report(y_test_balanced, y_pred_balanced))
    
    # Visualize confusion matrix
    sns.heatmap(confusion_matrix(y_test_balanced, y_pred_balanced), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


    # Decision Tree, SVM, and Random Forest performed best,
    # but i will consider Random Forest as it can handle complex data and is less prone to overfitting.






