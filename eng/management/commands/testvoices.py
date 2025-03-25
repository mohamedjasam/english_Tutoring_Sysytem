import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# List all available voices and print their names
voices = engine.getProperty('voices')
for voice in voices:
    print(voice.id, voice.name)  # Print the voice id and name

# Optionally, set the voice to Mark if it exists
for voice in voices:
    if "David" in voice.name:  # Check if the voice is "Mark"
        engine.setProperty('voice', voice.id)
        break
