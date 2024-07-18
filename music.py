from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
import librosa
import numpy as np

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

music_file_name = "spotifydown.com - Good Luck.mp3"

# Upload the file
music_file = genai.upload_file(path=music_file_name)

while music_file.state.name == "PROCESSING":
    print('_', end='')
    time.sleep(10)
    music_file = genai.get_file(music_file.name)
    
if music_file.state.name == "FAILED":
    raise ValueError(music_file.state.name)

file = genai.get_file(name=music_file.name)
print(f"Retrieved file '{file.display_name}' as: {music_file.uri}")

# Load the music file with librosa
y, sr = librosa.load(music_file_name, sr=None)

# Calculate energy using short-time Fourier transform (STFT)
hop_length = 512  # Number of samples between successive frames
frame_length = 2048  # Length of the frame for STFT
energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])

# Calculate the average energy for 30-second segments
segment_length = sr * 30  # 30 seconds in samples
num_segments = len(y) // segment_length

average_energy = [np.mean(energy[i*segment_length//hop_length:(i+1)*segment_length//hop_length]) for i in range(num_segments)]

# Find the segment with the highest average energy
max_energy_segment = np.argmax(average_energy)

# Extract the start and end time of the segment
start_time = max_energy_segment * 30  # in seconds
end_time = start_time + 30  # in seconds

# Convert times to minutes:seconds format
def seconds_to_minutes(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02}"

start_time_formatted = seconds_to_minutes(start_time)
end_time_formatted = seconds_to_minutes(end_time)

print(f"The most energetic 30-second segment is from {start_time_formatted} to {end_time_formatted} minutes.")

# Optionally, save this segment to a new file
y_segment = y[start_time*sr:end_time*sr]
librosa.output.write_wav('energetic_segment.wav', y_segment, sr)

# Now you can use this segment for further processing or analysis with the generative model
# Create the prompt.
prompt = "Describe this music."

# The Gemini 1.5 models are versatile and work with multimodal prompts
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Upload the new segment
segment_file_name = 'energetic_segment.wav'
segment_file = genai.upload_file(path=segment_file_name)

while segment_file.state.name == "PROCESSING":
    print('_', end='')
    time.sleep(10)
    segment_file = genai.get_file(segment_file.name)
    
if segment_file.state.name == "FAILED":
    raise ValueError(segment_file.state.name)

file = genai.get_file(name=segment_file.name)
print(f"Retrieved file '{file.display_name}' as: {segment_file.uri}")

response = model.generate_content([segment_file, prompt],
                                  request_options={"timeout": 600})

print(response.text)

