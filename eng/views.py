# import datetime
# import json
# from pyexpat.errors import messages
# from django.shortcuts import render,redirect,HttpResponse
# from .models import *

# from django.contrib import messages
# # Create your views here.
# def index(request):
#     return render(request, 'index.html')
# def chatbot(request):
#     return render(request,'chatbot.html')

# def talk_about_anything(request):
#     return render(request,'talk_about_anything.html')    

# def home(request):
#     if 'useremail' in request.session:
#         email = request.session['useremail']
#         try:
#             same = UserReg.objects.get(Email=email)
#             name = same.Name
#             return render(request, 'home.html',{'name': name}) 
#         except:
#             return redirect('login')
          
#     else:
#         return redirect('login')
# def RegisterUser(request):
#     if request.method == 'POST':
#         name = request.POST.get('name')
#         email = request.POST.get('email')   
#         password = request.POST.get('password')
#         phone = request.POST.get('phone') 
#         try:
#             same = UserReg.objects.get(Email=email)
#             if same:
#                 alert = "<script> alert('User Already exist'); window.location.href = '/register/';</script>"
#                 return HttpResponse(alert)

#         except:    
#             User = UserReg(Name=name,Email=email,Password=password,Phone=phone)
#             User.save()
#             return redirect('login')
#     else:
#         return render(request, 'register.html')

# def LoginUser(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')   
#         password = request.POST.get('password')
#         try:
#             check = UserReg.objects.get(Email=email,Password=password)
#             if check:
#                 request.session['useremail'] = check.Email
#                 return redirect('home') 
#         except Exception as e:
#             print(e)
#             alert = "<script> alert('invalid Username or password'); window.location.href = '/login/';</script>"      
#             return HttpResponse(alert)
#     else:
#          return render(request, 'login.html')
# def logout(request):
#     request.session.flush()
#     return redirect('index')
    
# def adlogin(request):
#     if request.method=='POST':
#         name = request.POST.get('name')
#         password = request.POST.get('password')
#         u = 'admin'
#         p = 'admin'
#         if name==u:
#             if password==p:
#                 return redirect('adhome')
#             else:
#              return render(request,"adlogin.html")
#         else:
#              return render(request,"adlogin.html")
#     else:
#          return render(request,"adlogin.html")
# def adhome(request):
#     return render(request,'adhome.html')
# def profile(request):
#     email = request.session.get('useremail')
    
#     if email is not None:
#         try:
#             user = UserReg.objects.get(Email=email)
#             return render(request, 'profile.html', {'user': user})
#         except UserReg.DoesNotExist:
#             messages.error(request, "User not found.")
#             return redirect('login')  
#     else:
#         messages.warning(request, "You need to log in to access your profile.")
#         return redirect('login') 
    
def editprofile(request):
    email = request.session.get('useremail') 
    try:
        user = UserReg.objects.get(Email=email)  # Ensure 'Email' matches the field name in the model
    except UserReg.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect('login')  # Redirect to login if user is not found

    if request.method == 'POST':
        # Get form data
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        level=request.POST.get('level')
        profile_pic= request.FILES.get('profile_pic')

        # Update user profile fields
        user.Name = name
        user.Email = email
        user.Phone = phone
        user.level=level
        
        # Update profile picture if provided
        if profile_pic:
            user.profile_pic = profile_pic

        # Save the updated User
        user.save()

        # Send success message
        messages.success(request, 'Profile updated successfully!')

        return redirect('profile')  

    return render(request, 'profile.html', {'user': user})

# # views.py
# # views.py
# from django.http import JsonResponse
# # from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# import pyttsx3
# import json
# # from django.http import JsonResponse
# # from django.views.decorators.csrf import csrf_exempt
# from django.conf import settings
# import os


def select_topic(request):
    return render(request, 'select_topic.html')


def practice_session(request):
    return render(request, 'practice_session.html')

def random(request):
    return render(request, 'random.html')

def report(request):
    return render(request,'report.html') 

def sample(request):
    return render(request,'sample.html')     
# def userlist(request):
#     user=UserReg.objects.all()
#     return render(request,'userlist.html',{'user':user})
# def deleteuser(request,id):
#     data=UserReg.objects.filter(id=id)
#     data.delete()
#     return redirect('userlist')


# from django.contrib import messages


# def feedback_rate(request):
#     if request.method == "POST":
#         feedback_text = request.POST.get('feedback_text')
#         rating = request.POST.get('rating')
        
#         if not feedback_text or not rating:
#             messages.error(request, "Please fill in all required fields.")
#             return render(request, 'feedback.html')

#         try:
#             rating = int(rating)
#             if rating not in [1, 2, 3, 4, 5]:
#                 raise ValueError("Invalid rating value")
#         except (ValueError, TypeError):
#             messages.error(request, "Invalid rating value. Please select a valid rating.")
#             return render(request, 'feedback.html')

#         # Create and save the Feedback instance
#         feedback = Feedback(
#             feedback_text=feedback_text,
#             rating=rating
#         )
#         feedback.save()

#         # Set a success message
#         messages.success(request, "Thank you for your feedback!")
#         return redirect('feedback')  # Redirect to refresh the form page and display the success message
    
#     # Render the feedback form
#     return render(request, 'feedback.html')

# def feedbacklist(request):
#     data=Feedback.objects.all()
#     return render(request,'feedbacklist.html',{'data':data})

# # import csv
# # import os
# # import json
# # from django.conf import settings
# # from django.http import JsonResponse
# # from django.views.decorators.csrf import csrf_exempt
# # import pyttsx3

# # # Load dataset at the start to avoid repeated I/O operations
# # data_path = os.path.join(settings.BASE_DIR, 'data.csv')
# # responses = {}

# # with open(data_path, 'r') as file:
# #     reader = csv.reader(file)
# #     next(reader)  # Skip header row
# #     for row in reader:
# #         request, response = row
# #         responses[request.lower()] = response

# # @csrf_exempt
# # def process_voice(request):
# #     if request.method == 'POST':
# #         try:
# #             # Get the message from the frontend
# #             data = json.loads(request.body)
# #             user_message = data.get('message', '').strip().lower()

# #             # Check for a matching response in the dataset
# #             bot_response = responses.get(user_message, "Sorry, I didn't understand that. Could you try asking another way?")

# #             # Generate speech (audio file)
# #             engine = pyttsx3.init()
# #             audio_file_name = "bot_response.mp3"
# #             audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file_name)

# #             # Ensure the directory exists, if not create it
# #             if not os.path.exists(settings.MEDIA_ROOT):
# #                 os.makedirs(settings.MEDIA_ROOT)

# #             # Save the speech to the file
# #             engine.save_to_file(bot_response, audio_file_path)
# #             engine.runAndWait()

# #             # Return the bot response and the audio URL
# #             audio_url = os.path.join(settings.MEDIA_URL, audio_file_name)
# #             return JsonResponse({
# #                 'response': bot_response,
# #                 'audio_url': audio_url
# #             })

# #         except Exception as e:
# #             print("Error in processing voice:", e)
# #             return JsonResponse({'response': 'An error occurred, please try again.'})


# import csv
# import os
# import json
# import pyttsx3
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from django.conf import settings


# import nltk
# # nltk.download('all')
# nltk.download('punkt')
# nltk.download('stopwords')

# # Initialize stemmer and stopwords
# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # Load the dataset into memory
# data_path = os.path.join(settings.BASE_DIR, 'data.csv')

# responses = {}

# with open(data_path, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip header row
#     for row in reader:
#         if len(row) == 3:  # Ensure the row has exactly 3 values
#             request, response, keywords = row
#             keywords = keywords.lower().split(', ')  # Convert keywords into a list
#             responses[request.lower()] = {
#                 'response': response,
#                 'keywords': keywords
#             }
#         else:
#             print(f"Skipping invalid row: {row}")

# # Tokenize and preprocess user input
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import string

# # Define preprocessing function
# # def preprocess_input(user_input):
# #     ps = PorterStemmer()
# #     custom_stop_words = set(stopwords.words("english")) - {"how", "you", "are" , "who"}
    
# #     # Tokenize and convert to lowercase
# #     tokens = word_tokenize(user_input.lower())
    
# #     # Remove punctuation and stopwords, then apply stemming
# #     filtered_tokens = [
# #         ps.stem(word) for word in tokens if word not in custom_stop_words and word not in string.punctuation
# #     ]
    
# #     return filtered_tokens
# def preprocess_input(user_input):
#     # Initialize stemmer and stopwords
#     custom_stop_words = set(stopwords.words("english")) - {"how", "you", "are", "who"}
    
#     # Tokenize and convert to lowercase
#     tokens = word_tokenize(user_input.lower())
    
#     # Remove punctuation and stopwords, then apply stemming
#     filtered_tokens = [
#         ps.stem(word) for word in tokens if word not in custom_stop_words and word not in string.punctuation
#     ]
    
#     return filtered_tokens


# # Function to find a match
# # def find_best_match(user_input):
# #     user_tokens = preprocess_input(user_input)
# #     print(f"Preprocessed input tokens: {user_tokens}") 
# #     best_match = None
# #     best_match_score = 0

# #     for request, data in responses.items():
# #         keyword_matches = 0
# #         for keyword in data['keywords']:
# #             if keyword in user_tokens:
# #                 keyword_matches += 1
# #         if keyword_matches > best_match_score:
# #             best_match_score = keyword_matches
# #             best_match = data['response']
    
# #     return best_match if best_match else "Sorry, I didn't understand that. Could you try asking another way?"

# import time
# audio_file_name = f"bot_response_{int(time.time())}.mp3"

# # def find_best_match(user_input):
# #     # First, check for direct keyword match (exact match)
# #     user_input_lower = user_input.lower()
# #     best_match = None
# #     best_match_score = 0

# #     for request, data in responses.items():
# #         # Check if any of the keywords directly match the user input
# #         for keyword in data['keywords']:
# #             if keyword in user_input_lower:
# #                 return data['response']  # If found a match, return the response immediately

# #     # If no direct match found, proceed with tokenizing and scoring
# #     user_tokens = preprocess_input(user_input)
# #     print(f"Preprocessed input tokens: {user_tokens}") 

# #     for request, data in responses.items():
# #         keyword_matches = 0
# #         for keyword in data['keywords']:
# #             if any(token in keyword for token in user_tokens):  # Partial match
# #                 keyword_matches += 1
# #         if keyword_matches > best_match_score:
# #             best_match_score = keyword_matches
# #             best_match = data['response']
    
# #     return best_match if best_match else "Sorry, I didn't understand that. Could you try asking another way?"
# from .nlp_utils import get_nlp_response  # Import the new NLP response function

# def find_best_match(user_input):
#     try:
#         # Use the NLP model to generate a response
#         bot_response = get_nlp_response(user_input)
#         return bot_response
#     except Exception as e:
#         print(f"Error generating NLP response: {e}")
#         # Fallback response in case of errors
#         return "Sorry, I encountered an error. Could you rephrase your question?"



# # @csrf_exempt
# # def process_voice(request):
# #     if request.method == 'POST':
# #         try:
# #             # Get the message from the frontend
# #             data = json.loads(request.body)
# #             user_message = data.get('message', '').strip()

# #             # Find the best match response
# #             bot_response = find_best_match(user_message)

# #             # Generate speech (audio file)
# #             engine = pyttsx3.init()
# #             audio_file_name = "bot_response.mp3"
# #             audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file_name)

# #             # Ensure the directory exists, if not create it
# #             if not os.path.exists(settings.MEDIA_ROOT):
# #                 os.makedirs(settings.MEDIA_ROOT)

# #             # Save the speech to the file
# #             engine.save_to_file(bot_response, audio_file_path)
# #             engine.runAndWait()

# #             # Return the bot response and the audio URL
# #             audio_url = os.path.join(settings.MEDIA_URL, audio_file_name)
# #             return JsonResponse({
# #                 'response': bot_response,
# #                 'audio_url': audio_url
# #             })

# #         except Exception as e:
# #             print("Error in processing voice:", e)
# #             return JsonResponse({'response': 'An error occurred, please try again.'})

# from .nlp_utils import get_nlp_response  # Import the NLP response function

# @csrf_exempt
# def process_voice(request):
#     if request.method == 'POST':
#         try:
#             # Get the message from the frontend
#             data = json.loads(request.body)
#             user_message = data.get('message', '').strip()

#             # Generate response using the NLP model
#             bot_response = get_nlp_response(user_message)

#             # Generate speech (audio file)
#             engine = pyttsx3.init()
#             audio_file_name = "bot_response.mp3"
#             audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file_name)

#             # Ensure the directory exists; create it if it doesn't exist
#             if not os.path.exists(settings.MEDIA_ROOT):
#                 os.makedirs(settings.MEDIA_ROOT)

#             # Save the bot response as an audio file
#             engine.save_to_file(bot_response, audio_file_path)
#             engine.runAndWait()

#             # Generate the public URL for the audio file
#             audio_url = os.path.join(settings.MEDIA_URL, audio_file_name)
#             return JsonResponse({
#                 'response': bot_response,
#                 'audio_url': audio_url
#             })

#         except Exception as e:
#             # Log the error and send a generic error response
#             print(f"Error in processing voice: {e}")
#             return JsonResponse({'response': 'An error occurred while processing your request. Please try again.'})
########


# Views
def index(request):
    return render(request, 'index.html')

def chatbot(request):
    return render(request, 'chatbot.html')
def chatbot1(request):
    return render(request, 'chatbot1.html')    

def talk_about_anything(request):
    return render(request, 'talk_about_anything.html')

def home(request):
    user = get_user_from_session(request)
    if user:
        return render(request, 'home.html', {'name': user.Name})
    return redirect('login')

def RegisterUser(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phone = request.POST.get('phone')

        if UserReg.objects.filter(Email=email).exists():
            messages.error(request, "User already exists.")
            return redirect('register')

        UserReg.objects.create(Name=name, Email=email, Password=password, Phone=phone)
        messages.success(request, "Registration successful!")
        return redirect('login')

    return render(request, 'register.html')

def LoginUser(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = UserReg.objects.filter(Email=email, Password=password).first()
        if user:
            request.session['useremail'] = user.Email
            return redirect('home')

        messages.error(request, "Invalid username or password.")
        return redirect('login')

    return render(request, 'login.html')

def logout(request):
    request.session.flush()
    return redirect('index')

def adlogin(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        if name == 'admin' and password == 'admin':
            return redirect('adhome')

        messages.error(request, "Invalid admin credentials.")
        return redirect('adlogin')

    return render(request, 'adlogin.html')

def adhome(request):
    return render(request, 'adhome.html')

def profile(request):
    user = get_user_from_session(request)
    if not user:
        messages.warning(request, "You need to log in to access your profile.")
        return redirect('login')

    if request.method == 'POST':
        user.Name = request.POST.get('name')
        user.Email = request.POST.get('email')
        user.Phone = request.POST.get('phone')
        user.level = request.POST.get('level')
        profile_pic = request.FILES.get('profile_pic')

        if profile_pic:
            user.profile_pic = profile_pic

        user.save()
        messages.success(request, "Profile updated successfully!")

    return render(request, 'profile.html', {'user': user})

def userlist(request):
    users = UserReg.objects.all()
    return render(request, 'userlist.html', {'users': users})

def deleteuser(request, id):
    UserReg.objects.filter(id=id).delete()
    messages.success(request, "User deleted successfully!")
    return redirect('userlist')

def feedback_rate(request):
    if request.method == 'POST':
        feedback_text = request.POST.get('feedback_text')
        rating = request.POST.get('rating')

        if not feedback_text or not rating or int(rating) not in [1, 2, 3, 4, 5]:
            messages.error(request, "Invalid input. Please fill all fields correctly.")
            return redirect('feedback')

        Feedback.objects.create(feedback_text=feedback_text, rating=rating)
        messages.success(request, "Thank you for your feedback!")
        return redirect('feedback')

    return render(request, 'feedback.html')

def feedbacklist(request):
    feedbacks = Feedback.objects.all()
    return render(request, 'feedbacklist.html', {'feedbacks': feedbacks})
    
def idioms(request):
    return render(request, 'idioms.html')    




#################
import os
import re
import time
import json
import csv
import string
import pyttsx3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from .models import UserReg, Feedback



import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Conversational Model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load CSV data
def load_csv(file_path):
    data = {}
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row["Question"].lower().strip()
            answer = row["Answer"].strip()
            data[question] = answer
    return data

csv_data = load_csv("data.csv")

# Bot Logic
def get_bot_response(user_input):
    user_input = user_input.lower().strip()

    # Step 1: CSV Match
    if user_input in responses:
        return responses[user_input]["response"]

    # Step 2: Transformer Model Response
    try:
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        bot_output = model.generate(
            inputs,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(bot_output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Transformer error: {e}")
        return "An error occurred while generating a response."

    # Step 3: Fallback
    return "Sorry, I didn't understand that. Could you try asking another way?"


# Testing Bot Responses
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     print(f"Bot: {get_bot_response(user_input)}")




# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize utilities
ps = PorterStemmer()
custom_stop_words = set(stopwords.words("english")) - {"how", "you", "are", "who"}

# Load dataset at server startup
DATA_PATH = os.path.join(settings.BASE_DIR, 'data.csv')
responses = {}

with open(DATA_PATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        if len(row) == 3:
            request, response, keywords = row
            responses[request.lower()] = {
                'response': response,
                'keywords': keywords.lower().split(', ')
            }

# Utility functions
def preprocess_input(user_input):
    tokens = word_tokenize(user_input.lower())
    filtered_tokens = [
        ps.stem(word) for word in tokens if word not in custom_stop_words and word not in string.punctuation
    ]
    return filtered_tokens

def get_user_from_session(request):
    email = request.session.get('useremail')
    return UserReg.objects.filter(Email=email).first()

def generate_audio_response(bot_response):
    audio_file_name = f"bot_response_{int(time.time())}.mp3"
    audio_file_path = os.path.join(settings.MEDIA_ROOT, audio_file_name)

    if not os.path.exists(settings.MEDIA_ROOT):
        os.makedirs(settings.MEDIA_ROOT)

    engine = pyttsx3.init()
    engine.save_to_file(bot_response, audio_file_path)
    engine.runAndWait()

    return os.path.join(settings.MEDIA_URL, audio_file_name)



@csrf_exempt
def process_voice(request):
    if request.method == 'POST':
        try:
            user_message = json.loads(request.body).get('message', '').strip()
            
            # Get bot response (either from CSV or model)
            bot_response = get_bot_response(user_message)

            # Generate audio URL for the response
            audio_url = generate_audio_response(bot_response)
            
            return JsonResponse({'response': bot_response, 'audio_url': audio_url})

        except Exception as e:
            print(f"Error processing voice: {e}")
            return JsonResponse({'response': "An error occurred, please try again."})



            # talk about movies section
import os
import json
import csv
import pyttsx3
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from groq import Groq

# Initialize Groq client
client = Groq(api_key="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")

# Directory for saving audio responses
AUDIO_DIR = os.path.join(settings.MEDIA_ROOT, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load CSV data into a dictionary
CSV_FILE_PATH = os.path.join(settings.BASE_DIR, 'data.csv')

def load_csv_data():
    data = {}
    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['Question'].lower().strip()
            answer = row['Answer'].strip()
            data[question] = answer
    return data

# Load the CSV data at server start
csv_data = load_csv_data()

# Find response in the CSV file
def find_response_in_csv(user_message):
    return csv_data.get(user_message.lower().strip(), None)

# Function to generate realistic TTS with pyttsx3
def generate_realistic_voice(text):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # List all available voices
    voices = engine.getProperty('voices')

    # Set the voice to Microsoft Mark
    for voice in voices:
        if "Mark" in voice.name:  # Check if the voice is "Mark"
            engine.setProperty('voice', voice.id)
            break

    # Adjust the rate (speed) and volume
    engine.setProperty('rate', 155)  # Speed
    engine.setProperty('volume', 1.0)  # Volume (1.0 is max)

    # Generate audio file name and path
    audio_file_name = f"response_{text[:10].replace(' ', '_')}_{len(text)}.mp3"
    audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)

    # Save the speech to an audio file
    engine.save_to_file(text, audio_file_path)
    engine.runAndWait()

    return audio_file_path

# Process Voice Request (with fallback to LLaMA model via Groq)
@csrf_exempt
def process_voice1(request):
    if request.method == "POST":
        try:
            # Step 1: Get user message
            data = json.loads(request.body)
            user_message = data.get("message")

            if not user_message:
                return JsonResponse({"error": "No message provided"}, status=400)

            # Step 2: Check the CSV file for a matching response
            csv_response = find_response_in_csv(user_message)

            if csv_response:
                bot_reply = csv_response
            else:
                # Fallback to LLaMA model using Groq API if no match is found
                chat_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": user_message}],
                    model="llama-3.3-70b-versatile"
                )
                bot_reply = chat_response.choices[0].message.content

            # Step 3: Generate realistic TTS response
            audio_file_path = generate_realistic_voice(bot_reply)

            # Step 4: Return the bot's response and audio file URL
            audio_url = os.path.join(settings.MEDIA_URL, "audio", os.path.basename(audio_file_path))
            return JsonResponse({"response": bot_reply, "audio_url": audio_url})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)
############

#idioms and phrases



# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# import requests  # Make sure this is installed via pip

# # API setup
# API_KEY = "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"
# API_URL = "https://api.groq.com/v1/models/llama-3.3-70b-versatile/completions"

# @csrf_exempt
# def generate_idioms_phrases(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             input_sentence = data.get("input_sentence", "").strip()

#             if not input_sentence:
#                 return JsonResponse({"error": "No input sentence provided"}, status=400)

#             # Modify the input sentence based on any context you want to add
#             prompt = "Generate an idiom or phrase that matches the following sentence:"

#             # Construct the message for the API
#             user_input_message = f"{prompt} {input_sentence}"

#             # Call the LLaMA model (or another model) via API
#             response = requests.post(API_URL, json={
#                 "messages": [{
#                     "role": "user",
#                     "content": user_input_message
#                 }],
#                 "model": "llama-3.3-70b-versatile",  # Specify the model to use
#                 "api_key": API_KEY,
#             })

#             # Check if the model's response is successful
#             if response.status_code == 200:
#                 model_data = response.json()
#                 idiom_or_phrase = model_data.get("choices", [{}])[0].get("message", {}).get("content", "No idiom or phrase generated.")
#             else:
#                 # Log the response status and content for debugging
#                 print(f"API response failed with status: {response.status_code}")
#                 print(f"Response content: {response.text}")
#                 idiom_or_phrase = f"Error generating idiom/phrase. Response: {response.text}"

#             # Return the idiom/phrase in the response
#             return JsonResponse({"idiom_or_phrase": idiom_or_phrase})

#         except requests.exceptions.RequestException as e:
#             # Log the exception for debugging
#             print(f"Request Exception: {str(e)}")
#             return JsonResponse({"error": f"Request failed: {str(e)}"}, status=500)
#         except Exception as e:
#             # General exception handling and logging
#             print(f"Unexpected error: {str(e)}")
#             return JsonResponse({"error": str(e)}, status=500)
    
#     return JsonResponse({"error": "Invalid request method"}, status=405)
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from groq import Groq

# Initialize the Groq client with a hardcoded API key
client = Groq(api_key="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")  # Replace with your actual API key

def generate_idioms_with_prompt(user_input):
    """
    Generate idioms or phrases related to the user input.
    """
    try:
        # Construct a clear and concise prompt
        prompt = (
            f"Given the sentence: '{user_input}', replace the part of the sentence that expresses emotion or state "
            "with a single idiom or phrase that has the same meaning. Return only the modified sentence, "
            "without any additional explanation. For example, if the input is 'I am happy because I passed the exam', "
            "the output should be 'I am over the moon because I passed the exam'."
        )
        # Query the Llama model with the prompt
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        # Extract and clean the response
        llama_response = chat_completion.choices[0].message.content.strip()
        return llama_response
    except Exception as e:
        # Raise an exception to be handled by the view
        raise Exception(f"Error generating idioms: {str(e)}")

@csrf_exempt
def generate_idioms_phrases(request):
    if request.method == "POST":
        try:
            # Parse the JSON body from the request
            data = json.loads(request.body)
            # Get user input from the JSON data
            user_message = data.get("message", "")
            # Handle empty input
            if not user_message:
                return JsonResponse({"error": "No message provided"}, status=400)

            # Generate idioms/phrases using prompt engineering
            idioms_and_phrases = generate_idioms_with_prompt(user_message)

            # Return the response as JSON
            return JsonResponse({
                "idioms_and_phrases": idioms_and_phrases,
            })
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=405)


########## eng/views.py




# views.py
# views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import GeneratedWord, CorrectionReport
from .utils import GrammarAnalyzer
import json
import random

# View for generating random words and rendering the random.html template
def generate_words_view(request):
    word_list = ["Movies: Talk about a movie that inspired you and why." , "Health: Explain the benefits of a balanced diet and regular exercise." ,"Books: Talk about a book that left a lasting impression on you." , "Hobbies: Explain your favorite hobby and how it benefits you","Education: Discuss the importance of lifelong learning."  , "Travel: Share your favorite travel destination and why you love it." , "Technology: Talk about how technology has changed communication in the past decade."]  # Example words
    selected_word = random.choice(word_list)
    word, created = GeneratedWord.objects.get_or_create(word=selected_word)
    request.session['current_word_id'] = word.id  # Store word ID in session
    return render(request, 'random.html', {'word': word})

# View for recording and transcribing user input
@csrf_exempt
def record_and_transcribe_view(request, word_id):
    word = get_object_or_404(GeneratedWord, id=word_id)
    if request.method == 'POST':
        transcription = request.POST.get('transcription', '').strip()
        if not transcription:
            return JsonResponse({'error': 'Transcription cannot be empty'}, status=400)

        # Save transcription and redirect to analysis
        word.transcription = transcription
        word.save()
        return redirect('analyze_transcription', word_id=word.id)
    
    return render(request, 'random.html', {'word': word})

# View for analyzing grammar and generating corrections
@csrf_exempt
def analyze_grammar_view(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request body
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data'}, status=400)

            # Validate that 'text' key exists in the request
            text = data.get('text', '').strip()
            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            # Initialize GrammarAnalyzer and process the text
            analyzer = GrammarAnalyzer()
            results = analyzer.analyze_text(text)

            # Return successful response with analysis results
            return JsonResponse({'grammar_analysis': results})

        except Exception as e:
            # Log the error for debugging purposes
            print(f"Error in analyze_grammar_view: {str(e)}")
            return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)

    # Handle invalid HTTP methods
    return JsonResponse({'error': 'Invalid request method'}, status=405)
# View for displaying the grammar analysis report
def analyze_transcription_view(request, word_id):
    word = get_object_or_404(GeneratedWord, id=word_id)
    transcription = word.transcription

    if not transcription:
        return render(request, 'report.html', {'error': 'No transcription available'})

    try:
        analyzer = GrammarAnalyzer()
        analysis_results = analyzer.analyze_text(transcription)
    except Exception as e:
        return render(request, 'report.html', {
            'error': f"Analysis failed: {str(e)}",
            'transcription': transcription
        })

    # Prepare data for the template
    return render(request, 'report.html', {
        'results': analysis_results,
        'transcription': transcription
    })
###interaction

# import os
# import json
# import random
# from django.http import JsonResponse
# from django.conf import settings
# from django.views.decorators.csrf import csrf_exempt
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from groq import Groq
# import pyttsx3

# # Initialize Groq client
# client = Groq(api_key="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")

# # Directory for saving audio responses
# AUDIO_DIR = os.path.join(settings.MEDIA_ROOT, "audio")
# os.makedirs(AUDIO_DIR, exist_ok=True)

# # Grammar correction model
# tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
# model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

# def correct_grammar(input_text):
#     inputs = tokenizer.encode(input_text, return_tensors='pt')
#     outputs = model.generate(inputs, max_length=1000, num_beams=5, early_stopping=True)
#     corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Post-process to ensure full grammatical correctness
#     if "went park" in corrected_text:
#         corrected_text = corrected_text.replace("went park", "went to the park")

#     return corrected_text

#     #positive feedback
# def generate_positive_reinforcement():
#     positive_responses = [
#         "Nicely done!",
#         "Perfect! That sounds much better.",
#         "Great job!",
#         "You got it!",
#         "Excellent!"
#     ]
#     return random.choice(positive_responses)    

# # Tutor feedback generation function
# def generate_tutor_feedback(original_message, corrected_message):
#     tutor_prompts = [
#         f"The correct way to say it is: \"{corrected_message}\". Try again!",
#         f"Let me help you with the correct version: \"{corrected_message}\". How does that sound?",
#         f"Try saying it like this: \"{corrected_message}\". Can you repeat that?",
#         f"Here's a better way: \"{corrected_message}\". Give it a go!",
#         f"You can say it better as: \"{corrected_message}\". Would you like to try?"
#     ]
#     chosen_prompt = random.choice(tutor_prompts)
    
#     # Generate response using Groq API
#     chat_response = client.chat.completions.create(
#         messages=[{"role": "user", "content": chosen_prompt}],
#         model="llama-3.3-70b-versatile"
#     )
#     bot_reply = chat_response.choices[0].message.content

#     # Ensure the feedback is concise
#     if len(bot_reply.split()) > 20:
#         bot_reply = bot_reply.split('.')[0] + "."

#     # Add a prompt for follow-up question generation
#     if "correct" in bot_reply.lower():
#         bot_reply += f" Almost! The correct way to say it is, \"{corrected_message}\". Try again!"

#     return bot_reply

# # Function to generate realistic TTS with pyttsx3
# def generate_realistic_voice(text):
#     engine = pyttsx3.init()
#     voices = engine.getProperty('voices')
#     for voice in voices:
#         if "Mark" in voice.name:  # Check if the voice is "Mark"
#             engine.setProperty('voice', voice.id)
#             break
#     engine.setProperty('rate', 155)  # Speed
#     engine.setProperty('volume', 1.0)  # Volume (1.0 is max)
#     audio_file_name = f"response_{text[:10].replace(' ', '')}{len(text)}.mp3"
#     audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
#     engine.save_to_file(text, audio_file_path)
#     engine.runAndWait()
#     return audio_file_path

# def generate_follow_up_question(context):
#     FOLLOW_UP_PROMPT = """
#     You are an English language tutor helping a user practice conversational English. Based on the user's input, generate a SHORT and engaging follow-up question (no more than 8 words). Keep it simple and natural.
#     User Input: {context}
#     Follow-up Question:
#     """
#     prompt = FOLLOW_UP_PROMPT.format(context=context)
    
#     chat_response = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile"
#     )
#     follow_up_question = chat_response.choices[0].message.content

#     # Ensure the follow-up question is concise
#     if len(follow_up_question.split()) > 8:
#         follow_up_question = follow_up_question.split('?')[0][:50] + "?"
    
#     return follow_up_question

#     # Ensure the follow-up question is concise
#     if len(follow_up_question.split()) > 10:
#         follow_up_question = follow_up_question.split('?')[0] + "?"
    
#     return follow_up_question

# def handle_short_response(user_message, topic):
#     PROMPT_FOR_SHORT_RESPONSE = """
#     You are an English language tutor helping a user improve their conversational English. The user has provided a short response: "{input}". 
#     Suggest a better or more complete way to phrase this response based on the context of the conversation.
#     If no context is available, provide a general suggestion for expanding the sentence.
#     Enhanced Phrase:
#     """
#     prompt = PROMPT_FOR_SHORT_RESPONSE.format(input=user_message)
    
#     chat_response = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile"
#     )
#     enhanced_phrase = chat_response.choices[0].message.content

#     # Ensure the enhanced phrase is concise
#     if len(enhanced_phrase.split()) > 20:
#         enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
#     return enhanced_phrase

# def handle_vague_response(user_message, topic):
#     PROMPT_FOR_VAGUE_RESPONSE = """
#     You are an English language tutor helping a user improve their conversational English. The user has provided a vague or incomplete response: "{input}".
#     Suggest a better or more complete way to phrase this response in a CASUAL and FRIENDLY tone. Keep the response short and natural.
#     If no context is available, provide a general suggestion for expanding the sentence.
#     Enhanced Phrase:
#     """
#     prompt = PROMPT_FOR_VAGUE_RESPONSE.format(input=user_message)
    
#     chat_response = client.chat.completions.create(
#         messages=[{"role": "user", "content": prompt}],
#         model="llama-3.3-70b-versatile"
#     )
#     enhanced_phrase = chat_response.choices[0].message.content

#     # Ensure the enhanced phrase is concise
#     if len(enhanced_phrase.split()) > 20:
#         enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
#     # Add a suggestion for the user to try again
#     bot_reply = f"Try saying this: \"{enhanced_phrase}\". Can you repeat it?"
    
#     return bot_reply

#     # Ensure the enhanced phrase is concise
#     if len(enhanced_phrase.split()) > 20:
#         enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
#     return enhanced_phrase

#     # Ensure the enhanced phrase is concise
#     if len(enhanced_phrase.split()) > 20:
#         enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
#     return enhanced_phrase
# @csrf_exempt
# def process_voice_dynamic(request, topic):
#     if request.method == "POST":
#         try:
#             # Step 1: Get user message
#             data = json.loads(request.body)
#             user_message = data.get("message")
#             if not user_message:
#                 return JsonResponse({"error": "No message provided"}, status=400)

#             # Step 2: Check if this is the first interaction for "Talk About Anything"
#             if topic == "anything":
#                 # Use session or database to track conversation state (optional)
#                 if not hasattr(request, 'session'):
#                     request.session = {}

#                 if not request.session.get('initial_greeting_done', False):
#                     # First interaction: Ask "How are you doing today?"
#                     bot_reply = "Hello! How are you doing today?"
#                     request.session['initial_greeting_done'] = True
#                     request.session.save()

#                     # Generate TTS audio
#                     audio_file_path = generate_realistic_voice(bot_reply)
#                     audio_url = os.path.join(settings.MEDIA_URL, "audio", os.path.basename(audio_file_path))
#                     return JsonResponse({"response": bot_reply, "audio_url": audio_url})

#             # Step 3: Correct grammar if necessary
#             corrected_message = correct_grammar(user_message)

#             print(f"User Message: {user_message}")
#             print(f"Corrected Message: {corrected_message}")

#             # Step 4: Define topic-specific prompts
#             if topic == "strangers":
#                 TUTOR_PROMPT = """
#                 You are an English language tutor focused on conversations with strangers. Your role is to help users improve their English while practicing introductions and small talk.
#                 When given a sentence, provide a response that includes:
#                 1. A confirmation that the user has made progress if they correct their mistake.
#                 2. Suggestions for alternative or better ways to phrase the sentence.
#                 3. Encouragement and follow-up questions about starting conversations to maintain the flow.
#                 """
#             elif topic == "movies":
#                 TUTOR_PROMPT = """
#                 You are an English language tutor focused on conversations about movies. Your role is to help users improve their English while discussing films.
#                 When given a sentence, provide a response that includes:
#                 1. A confirmation that the user has made progress if they correct their mistake.
#                 2. Suggestions for alternative or better ways to phrase the sentence.
#                 3. Encouragement and follow-up questions about movies to maintain the conversation.
#                 """
#             elif topic == "hobbies":
#                 TUTOR_PROMPT = """
#                 You are an English language tutor focused on conversations about hobbies. Your role is to help users improve their English while discussing their interests.
#                 When given a sentence, provide a response that includes:
#                 1. A confirmation that the user has made progress if they correct their mistake.
#                 2. Suggestions for alternative or better ways to phrase the sentence.
#                 3. Encouragement and follow-up questions about hobbies to maintain the conversation.
#                 """
#             else:  # Default to general topic ("Talk About Anything")
#                 TUTOR_PROMPT = """
#                 You are an English language tutor focused on casual conversations. Your role is to help users practice conversational English by discussing everyday topics.
#                 When given a sentence, provide a response that includes:
#                 1. A confirmation that the user has made progress if they correct their mistake.
#                 2. Suggestions for alternative or better ways to phrase the sentence.
#                 3. Encouragement and follow-up questions to maintain the conversation.
#                 """

#             # Step 5: Handle vague or short responses
#             if len(user_message.split()) < 4 or "don't know" in user_message.lower():
#                 bot_reply = handle_vague_response(user_message, topic)
#             else:
#                 # Step 6: Check if the user needs grammar correction
#                 if corrected_message != user_message:
#                     bot_reply = generate_tutor_feedback(user_message, corrected_message)

#                     # If the user repeats the same message, check if it matches the corrected version
#                     if user_message.lower().strip() == corrected_message.lower().strip():
#                         bot_reply = generate_positive_reinforcement()
#                         # Dynamically generate follow-up question
#                         bot_reply += " " + generate_follow_up_question(corrected_message)
#                     else:
#                         # If the user didn't correct their mistake, ensure the bot reply contains the suggestion
#                         if "try again" not in bot_reply.lower():
#                             bot_reply += f" The correct way to say it is, \"{corrected_message}\". Try again!"
#                 else:
#                     # If no grammar error, generate positive reinforcement and ask a follow-up question
#                     bot_reply = generate_positive_reinforcement()
#                     # Dynamically generate follow-up question
#                     bot_reply += " " + generate_follow_up_question(user_message)

#             # Step 7: Generate TTS audio
#             audio_file_path = generate_realistic_voice(bot_reply)
#             audio_url = os.path.join(settings.MEDIA_URL, "audio", os.path.basename(audio_file_path))

#             # Step 8: Return the response
#             return JsonResponse({"response": bot_reply, "audio_url": audio_url})

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#     return JsonResponse({"error": "Invalid request method."}, status=405)

###interaction
import os
import json
import random
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from groq import Groq
import pyttsx3

# Initialize Groq client with hardcoded API key
client = Groq(api_key="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")

# Directory for saving audio responses
AUDIO_DIR = os.path.join(settings.MEDIA_ROOT, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Grammar correction model
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

def correct_grammar(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=1000, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process to ensure full grammatical correctness
    if "went park" in corrected_text:
        corrected_text = corrected_text.replace("went park", "went to the park")
    
    return corrected_text

def generate_positive_reinforcement():
    positive_responses = [
        "Nicely done!", 
        "Perfect! That sounds much better.", 
        "Great job!", 
        "You got it!",
        "Excellent!"
    ]
    return random.choice(positive_responses)

def generate_tutor_feedback(original_message, corrected_message, topic_prompt):
    tutor_prompts = [
        f"The correct way to say it is: \"{corrected_message}\". Try again!",
        f"Let me help you with the correct version: \"{corrected_message}\". How does that sound?",
        f"Try saying it like this: \"{corrected_message}\". Can you repeat that?",
        f"Here's a better way: \"{corrected_message}\". Give it a go!",
        f"You can say it better as: \"{corrected_message}\". Would you like to try?"
    ]
    
    chosen_prompt = random.choice(tutor_prompts)
    
    # Generate response using Groq API with topic-specific prompt
    chat_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": topic_prompt},
            {"role": "user", "content": chosen_prompt}
        ],
        model="llama-3.3-70b-versatile"
    )
    bot_reply = chat_response.choices[0].message.content
    
    # Ensure the feedback is concise
    if len(bot_reply.split()) > 20:
        bot_reply = bot_reply.split('.')[0] + "."
    
    # Add a prompt for follow-up question generation
    if "correct" in bot_reply.lower():
        bot_reply += f" Almost! The correct way to say it is: \"{corrected_message}\". Try again!"
    
    return bot_reply

def generate_realistic_voice(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # Improved voice selection with fallback
    selected_voice = None
    for voice in voices:
        if "Mark" in voice.name:
            selected_voice = voice.id
            break
    if not selected_voice:
        selected_voice = voices[0].id  # Fallback to first available voice
    
    engine.setProperty('voice', selected_voice)
    engine.setProperty('rate', 155)  # Speed
    engine.setProperty('volume', 1.0)  # Volume (1.0 is max)
    
    # Sanitized filename
    safe_text = "".join(c if c.isalnum() else "_" for c in text[:10])
    audio_file_name = f"response_{safe_text}_{len(text)}.mp3"
    audio_file_path = os.path.join(AUDIO_DIR, audio_file_name)
    
    engine.save_to_file(text, audio_file_path)
    engine.runAndWait()
    return audio_file_path

def generate_follow_up_question(context):
    FOLLOW_UP_PROMPT = """You are an English language tutor helping a user practice conversational English. 
    Based on the user's input, generate a SHORT and engaging follow-up question (no more than 8 words). 
    Keep it simple and natural.\n\nUser Input: {context}\nFollow-up Question:"""
    prompt = FOLLOW_UP_PROMPT.format(context=context)
    
    chat_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    follow_up_question = chat_response.choices[0].message.content
    
    # Ensure conciseness
    parts = follow_up_question.split('?')
    if len(parts[0].split()) > 8:
        follow_up_question = parts[0][:50] + "?"
    return follow_up_question

def handle_short_response(user_message, topic):
    PROMPT = """You are an English language tutor helping a user improve their conversational English. 
    The user has provided a short response: "{input}". Suggest a better or more complete way to phrase 
    this response. If no context is available, provide a general suggestion for expanding the sentence."""
    prompt = PROMPT.format(input=user_message)
    
    chat_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    enhanced_phrase = chat_response.choices[0].message.content
    
    # Truncate if too long
    if len(enhanced_phrase.split()) > 20:
        enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
    return enhanced_phrase

def handle_vague_response(user_message, topic):
    PROMPT = """You are an English language tutor helping a user improve their conversational English. 
    The user has provided a vague response: "{input}". Suggest a better way to phrase this in a 
    CASUAL and FRIENDLY tone. Keep the response short and natural."""
    prompt = PROMPT.format(input=user_message)
    
    chat_response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    enhanced_phrase = chat_response.choices[0].message.content
    
    # Truncate if too long
    if len(enhanced_phrase.split()) > 20:
        enhanced_phrase = enhanced_phrase.split('.')[0] + "."
    
    return f"Try saying this: \"{enhanced_phrase}\". Can you repeat it?"

@csrf_exempt
def process_voice_dynamic(request, topic):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message")
            if not user_message:
                return JsonResponse({"error": "No message provided"}, status=400)
            
            # Handle "Talk About Anything" initial greeting
            if topic == "anything":
                if not hasattr(request, 'session'):
                    request.session = {}
                
                if not request.session.get('initial_greeting_done', False):
                    bot_reply = "Hello! How are you doing today?"
                    request.session['initial_greeting_done'] = True
                    request.session.save()
                    audio_path = generate_realistic_voice(bot_reply)
                    audio_url = os.path.join(settings.MEDIA_URL, "audio", os.path.basename(audio_path))
                    return JsonResponse({"response": bot_reply, "audio_url": audio_url})
            
            corrected_message = correct_grammar(user_message)
            
            # Topic-specific prompts
            if topic == "strangers":
                topic_prompt = """You are a tutor focusing on conversations with strangers. 
                Provide feedback that includes: confirmation of progress, phrasing suggestions, 
                and follow-up questions about starting conversations."""
            elif topic == "movies":
                topic_prompt = """You are a tutor focusing on movie conversations. 
                Provide feedback with: progress confirmation, phrasing suggestions, 
                and movie-related follow-up questions."""
            elif topic == "hobbies":
                topic_prompt = """You are a tutor focusing on hobby discussions. 
                Provide feedback with: progress confirmation, phrasing suggestions, 
                and hobby-related follow-up questions."""
            else:
                topic_prompt = """You are a general conversation tutor. 
                Provide feedback with: progress confirmation, phrasing suggestions, 
                and general follow-up questions."""
            
            # Handle short/vague responses
            if len(user_message.split()) < 4 or "don't know" in user_message.lower():
                bot_reply = handle_vague_response(user_message, topic)
            else:
                if corrected_message != user_message:
                    bot_reply = generate_tutor_feedback(user_message, corrected_message, topic_prompt)
                    
                    # Check if user repeated the same message
                    if user_message.lower().strip() == corrected_message.lower().strip():
                        bot_reply = generate_positive_reinforcement()
                else:
                    bot_reply = generate_positive_reinforcement()
                
                # Add follow-up question
                bot_reply += " " + generate_follow_up_question(corrected_message)
            
            audio_path = generate_realistic_voice(bot_reply)
            audio_url = os.path.join(settings.MEDIA_URL, "audio", os.path.basename(audio_path))
            
            return JsonResponse({"response": bot_reply, "audio_url": audio_url})
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid request method."}, status=405)