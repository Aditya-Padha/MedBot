import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

hospital_data = pd.read_csv("HospitalsInIndia.csv")

hospital_data['Location'] = hospital_data['State'] + ' ' + hospital_data['City'] + ' ' + hospital_data['Pincode'].astype(str)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(hospital_data['Location'])

def recommend_hospital(state, city, pincode):
    input_location = state + ' ' + city + ' ' + str(pincode)

    input_tfidf = tfidf_vectorizer.transform([input_location])

    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)

    most_similar_index = similarity_scores.argmax()

    recommended_hospital = hospital_data.iloc[most_similar_index]

    hospital_dict = {
        'Hospital': recommended_hospital['Hospital'],
        'State': recommended_hospital['State'],
        'City': recommended_hospital['City'],
        'Pincode': recommended_hospital['Pincode']
    }

    return hospital_dict
