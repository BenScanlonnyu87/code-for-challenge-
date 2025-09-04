This is a script that demonstrates a simple medical chatbot using NLP and machine learning written in python.  
I choese python due to its ability to use pandas to first clean and normailize the data from the xecell spreadsheet as 
import pandas as pd


Required Libaries 
pip install scikit-learn nltk
# python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np



mport pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('mport pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('your_file_name.xlsx', sheet_name='Sheet1')
.xlsx', sheet_name='Sheet1')

intents_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey", "Good morning", "Good afternoon"],
            "responses": ["Hello! How can I help you with your health today?", "Hi there. What can I do for you?", "Greetings! How are you feeling?"]
        },
        {
            "tag": "symptoms_cough",
            "patterns": ["I have a cough", "My throat is scratchy", "I'm coughing a lot"],
            "responses": ["Coughing can be a symptom of many things. Is it a dry or productive cough?", "Please describe your cough. Is it constant or occasional?"]
        },
        {
            "tag": "symptoms_fever",
            "patterns": ["I have a fever", "My temperature is high", "I feel hot"],
            "responses": ["A fever can be a sign of infection. Do you have any other symptoms like a headache or body aches?", "How high is your temperature?"]
        },
        {
            "tag": "symptoms_headache",
            "patterns": ["I have a headache", "My head hurts", "I'm getting a migraine"],
            "responses": ["Headaches can have many causes. Can you tell me if it's a dull ache or sharp pain?", "Is the pain on one side of your head or all over?"]
        },
        {
            "tag": "medication_info",
            "patterns": ["What can I take for a headache?", "What medicine for fever?", "Cough drops"],
            "responses": ["I can't recommend specific medication. Please consult a doctor or pharmacist.", "It's best to speak with a healthcare professional about medication."]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Thanks, bye"],
            "responses": ["Goodbye! Take care and feel better.", "Farewell! If you need anything else, just ask.", "Stay healthy!"]
        },
        {
            "tag": "default",
            "patterns": [],
            "responses": ["I'm sorry, I don't understand. Could you please rephrase that?", "Please ask me a question about health or symptoms.", "I am not a medical professional. For serious issues, please consult a doctor."]
        }
    ]
}

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the data
def preprocess_data():
    """Extracts patterns and tags from the intents data for training."""
    patterns = []
    tags = []
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    return patterns, tags

# Tokenize and lemmatize a sentence
def process_sentence(sentence):
    """Tokenizes and lemmatizes a sentence for model input."""
    words = nltk.word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Training the model
def train_model():
    patterns, tags = preprocess_data()
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the patterns
    X = vectorizer.fit_transform(patterns)
    y = tags
    
    # Train a LinearSVC model
    model = LinearSVC()
    model.fit(X, y)
    
    return model, vectorizer
	# Get a response from the chatbot
	def get_response(user_input, model, vectorizer):
		"""Predicts the intent and returns a random response."""
		# Preprocess the user input
		processed_input = process_sentence(user_input)
		
		# Transform the input using the same vectorizer
		X_input = vectorizer.transform([processed_input])
		
		# Predict the intent
		predicted_tag = model.predict(X_input)[0]
		
		# Get the corresponding responses
		for intent in intents_data['intents']:
			if intent['tag'] == predicted_tag:
				return random.choice(intent['responses'])

				Fallback to default if no match is found (shouldn't happen with LinearSVC)
				return random.choice(intents_data['intents'][-1]['responses'])

			
if __name__ == "__main__":
    print("Welcome to the simple medical chatbot. How can I help you? (type 'quit' to exit)")
    
    # Train the model once at the start
    try:
        model, vectorizer = train_model()
    except Exception as e:
        print(f"Error initializing the chatbot: {e}")
        print("Please ensure you have installed all dependencies and downloaded NLTK corpora.")
        exit()

		while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye! Stay healthy.")
            break
        
        response = get_response(user_input, model, vectorizer)
        print(f"Chatbot: {response}")

		while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye! Stay healthy.")
            break
        
        response = get_response(user_input, model, vectorizer)
        print(f"Chatbot: {response}")


					
