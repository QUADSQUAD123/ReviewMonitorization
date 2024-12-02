import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_excel(r"C:\Users\LENOVO\Downloads\enhanced_synthetic_reviews.xlsx")

X = df['review_text']
y = df['is_fake']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

df['predicted_is_fake'] = model.predict(X_vectorized)

genuine_reviews = df[df['predicted_is_fake'] == 0]
print("\nGenuine Reviews Removed:")
print(genuine_reviews[['sno', 'review_text', 'rating', 'category', 'predicted_is_fake']])

df_cleaned = df[df['predicted_is_fake'] == 1]

df_cleaned.to_excel(r"C:\Users\LENOVO\Desktop\cleaned_reviews_with_category.xlsx", index=False)
print("\nCleaned dataset saved as 'cleaned_reviews_with_category.xlsx'.")

before_filtering = df['is_fake'].value_counts()
after_filtering = df_cleaned['predicted_is_fake'].value_counts()

plt.figure(figsize=(10, 6))
plt.plot(before_filtering.index, before_filtering.values, label="Before Filtering", marker='o', color='blue')
plt.plot(after_filtering.index, after_filtering.values, label="After Filtering", marker='o', color='red')

plt.xlabel('Review Type (0: Genuine, 1: Fake)')
plt.ylabel('Number of Reviews')
plt.title('Comparison of Fake vs Genuine Reviews (Before and After Filtering)')
plt.legend()

plt.grid(True)
plt.show()

print("\nCleaned Dataset (Fake Reviews Only):")
print(df_cleaned[['sno', 'review_text', 'rating', 'category', 'predicted_is_fake']])
print("File saved at: C:\\Users\\LENOVO\\Desktop\\cleaned_reviews_with_category.xlsx")
