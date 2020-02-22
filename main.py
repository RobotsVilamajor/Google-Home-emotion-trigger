from keras.models import model_from_json
from googlehomepush import GoogleHome
from googlehomepush.http_server import serve_file
import numpy as np
import os
import glob
import cv2
import pygame
import time
import configparser
import ast
import random
import argparse

argpar = argparse.ArgumentParser()
argpar.add_argument('song', nargs='?', default=0, type=int)

args = argpar.parse_args()

print(args.song)

# Open webcam
cam = cv2.VideoCapture(0)

# Initialize pygame window
pygame.init()

window = pygame.display.set_mode((1000, 500), pygame.RESIZABLE)

# Create face detector
casc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Open configuration file
config = configparser.ConfigParser()
config.read("config.cfg")
config = config["DATA"]

# Get config values
EMOTIONS =	ast.literal_eval(config["emotions"])
COLORS =	ast.literal_eval(config["colors"])
WEIGHTS =	ast.literal_eval(config["weights"])
MUSIC =		[glob.glob(f"Music\\{x}\\*.mp3") for x in range(6)]
HOST =		config["host"]
play_song =	args.song != 0 #config.getboolean("play_song")
path =		config["model"]

print(play_song)

# Load keras model
json_file = open(f"{path}.json")
file = json_file.read()
json_file.close()

model = model_from_json(file)
model.load_weights(f"{path}.json.h5")

# Connect to google home
device = GoogleHome(host=HOST)

# Timer start time
start_time = time.time()

# History of emotions
history = []

# Main loop
run = True
while run:
	
	# Get user signal to close pygame window
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
			pygame.quit()
	
	# Read webcam image
	opened, img = cam.read()
	
	# Correct webcam image color
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Get webcam image size
	h, w, c = img.shape
	img_size = (w, h)
	
	# Detect faces in the image
	faces = casc.detectMultiScale(img, 1.05)
	
	# For each face
	for (x, y, w, h) in faces:
		
		# Crop the face
		croped = img[y : y + h, x : x + w]
		croped = cv2.resize(croped, (128, 128))
		croped = cv2.cvtColor(croped, cv2.COLOR_RGB2GRAY)
		
		# Model predict emotion
		out = model.predict(croped.reshape((1, 128, 128, 1)))[0]
		
		# Get emotion index
		element = out.argmax()
		
		# Save emotion index
		history.append(element)
		
		# Draw rectangle on face with appropiate color
		cv2.rectangle(img, (x, y), (x + w, y + h), COLORS[out.argmax()], 2)
	
	
	# If delay time has elapsed
	if time.time() - start_time > int(config["delay"]):
		
		# Restart timer
		start_time = time.time()
		
		# Create array of weighted counts of emotions
		counts = [history.count(x) * WEIGHTS[x] for x in range(6)]
		
		# Check if not empty
		if sum(counts) != 0:
			
			# Get the highest count emotion
			result = np.array(counts).argmax()
			
			# Get the emotion
			emotion = EMOTIONS[result]
			
			# Calculate emotion percent
			percent = int((counts[result] / sum(counts)) * 100)
			
			print(play_song)
			
			# If play song is true
			if play_song:
				
				# Get random song from path
				music = random.choice(MUSIC[result])
				
				# Create song server for google home
				song = serve_file(music)
				
				# Get song name
				name = os.path.splitext(os.path.basename(music))[0]
				
				# Make google home talk
				sentence = config["sentence_song"].format(percent=percent, emotion=emotion, song=name)
				print(sentence)
				device.say(sentence, "es")
				
				# Wait for google home to finish the previous sentence
				time.sleep(10)
				
				# Play the song
				device.play(song)
				
				# Exit program
				run = False
				pygame.quit()
			
			# If play song is false
			else:
				
				# Make google home talk
				sentence = config["sentence_no_song"].format(percent=percent, emotion=emotion)
				print(sentence)
				device.say(sentence, "es")
				
			# Clear stored emotions
			history.clear()
			
	# Draw webcam image into pygame window
	surf = pygame.image.frombuffer(img.tostring(), img_size, 'RGB')
	window.blit(surf, (0, 0))
	
	# Update pygame window
	pygame.display.update()
	
	# cv2.waitKey(0)