#import pandas as pd
# Load the dataset
#doctors_df = pd.read_csv('doctors.csv')


#def recommend_doctorsz(specialization, city, state):
    # Convert input to lowercase for case-insensitive comparison
#    specialization = specialization.lower()
#    city = city.lower()
 #   state = state.lower()

  #  # Convert dataframe columns to lowercase for case-insensitive comparison
   # doctors_df['specialization'] = doctors_df['specialization'].str.lower()
    #doctors_df['city'] = doctors_df['city'].str.lower()
    #doctors_df['state'] = doctors_df['state'].str.lower()

    # Filter the dataset based on provided criteria
    #filtered_doctors = doctors_df[(doctors_df['specialization'] == specialization) & (doctors_df['city'] == city) & (doctors_df['state'] == state)]

    #if filtered_doctors.empty:
        #return {"message": "Sorry, no doctors found matching the provided criteria."}
    #else:
        #recommended_doctors = filtered_doctors.to_dict(orient='records')
        #return {"Recommended_Doctors": recommended_doctors}


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doctors_data = pd.read_csv("doctors (2).csv")

doctors_data['SearchKey'] = doctors_data['specialization'] + ' ' + doctors_data['city'] + ' ' + doctors_data['state']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(doctors_data['SearchKey'])

def recommend_doctor(specialization, city, state):
    input_search_key = specialization.lower() + ' ' + city.lower() + ' ' + state.lower()

    input_tfidf = tfidf_vectorizer.transform([input_search_key])

    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)

    # Handling the case where no doctors match the input search key
    if similarity_scores.max() == 0:
        return {"message": "Sorry, no doctors found matching the provided criteria."}

    most_similar_index = similarity_scores.argmax()

    recommended_doctor = doctors_data.iloc[most_similar_index]

    doctor_dict = {
        'Doctor': recommended_doctor['doctor'],
        'Specialization': recommended_doctor['specialization'],
        'City': recommended_doctor['city'],
        'State': recommended_doctor['state'],
        'Address': recommended_doctor['address'],
        'Link': recommended_doctor['link']
    }
    print(doctor_dict)
    return {"Recommended_Doctors": doctor_dict}



