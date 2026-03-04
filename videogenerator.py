from moviepy import TextClip, VideoFileClip, CompositeVideoClip, ColorClip, ImageClip
import numpy as np
import cv2
import random

height, width = 108, 192
seed = 42

duration = 5	# Duration of the animation in seconds
fps = 10		# Frames per second

halfstep = 1	# How many seconds should it take before a new pattern starts to show
step = 2 * halfstep

patterns = int(duration / halfstep)

# The pattern names followed by frequency and number of params
pattern = [
	["+", 6, 2],		# blending = smooth gradients
	["*", 1, 2],		# rare contrast spikes
	["pow", 1, 3],		# rare intensity shaping
	["sin", 8, 1],		# waves are king
	["cos", 8, 1],		# phase-shifted waves
	["inv", 2, 1],		# occasional color inversion for motion
	["smooth", 4, 1],	# smoothstep (soft contrast curve)
	["mix", 4, 3],		# lerp blend (weighted blend instead of average)
	["abs", 4, 1],		# mirrored waves, great for ocean ripples
]

# The base parameters and frequencies
baseInfo = [
	["x", 3],
	["y", 3],
	["r", 3],
	["t", 2],
	["0", 1], 
	["1", 3]
]

# Calculate total frequencies for efficiency
total = 0
for p in pattern:
	total += p[1]

bi = 0
for p in baseInfo:
	bi += p[1]

# Select a weighted random element from list l of size t, where the weights are element 1
def selector(l, t):
	r = int(random.random() * t)
	i = -1
	while r >= 0:
		i += 1
		r -= l[i][1]

	return l[i]

class Pattern(object):
	# The args vary from either patterns or random numbers depending on the name of the operation
	def __init__(self, name, *args):
		self.name = name
		self.args = args

	# When this pattern is applied on the X, Y and T, it will return the color of that pixel or pixels
	def apply(self, x, y, t):
		global baseInfo, bi

		if self.name == "base":							# Gives a raw pixel value based on either x, y, or t
			l = []
			r = np.sqrt(2 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
			d = {
				"x": x,
				"y": y,
				"r": r,
				"t": t,
				"0": t * 0,								# t * 0 because we want an np.zeros of size t
				"1": np.ones_like(t)
			}

			for a in self.args[0]:
				l.append(d[a])
			return np.array(l)

		elif self.name == "+":							# Averages two patterns
			return (self.args[0].apply(x, y, t) + self.args[1].apply(x, y, t)) / 2
		elif self.name == "*":							# Multiply two patterns
			return self.args[0].apply(x, y, t) * self.args[1].apply(x, y, t)
		elif self.name == "pow":						# Takes one pattern and takes it to the power of the division of two other patterns
			a2 = self.args[2].apply(x, y, t)
			if np.sum(a2) == 0:
				return a2
			else:
				return np.pow(self.args[0].apply(x, y, t), np.sum(self.args[1].apply(x, y, t)) / np.sum(a2))
		elif self.name == "sin":						# The sine of a pattern
			return (np.sin((self.args[0].apply(x, y, t) + x * 0.5) * 2 * np.pi) + 1) / 2
		elif self.name == "cos":						# The cosine of a pattern
			return (np.cos((self.args[0].apply(x, y, t) + x * 0.5) * 2 * np.pi) + 1) / 2
		elif self.name == "inv":						# The linear inverse of a pattern
			return 1 - self.args[0].apply(x, y, t)
		elif self.name == "smooth":
			v = self.args[0].apply(x, y, t)
			return v * v * (3 - 2 * v)
		elif self.name == "mix":
			a = self.args[0].apply(x, y, t)
			b = self.args[1].apply(x, y, t)
			w = self.args[2].apply(x, y, t)
			return a * w + b * (1 - w)
		elif self.name == "mix":
			a = self.args[0].apply(x, y, t)
			b = self.args[1].apply(x, y, t)
			w = self.args[2].apply(x, y, t)
			return a * w + b * (1 - w)
		elif self.name == "abs":
			return np.abs(self.args[0].apply(x, y, t) - 0.5) * 2

# Creates a pattern of length max_len
def getPattern(max_len):
	global pattern, total

	if max_len == 0:
		l = []
		for i in range(3):
			l.append(selector(baseInfo, bi)[0])

		return Pattern("base", l)
	else:
		choice = selector(pattern, total)

		things = []
		for a in range(choice[2]):
			things.append(getPattern(max_len - 1))

		return Pattern(choice[0], *things)

# Gives a random value that will be used for depth. R varies from 0 to 1 and the returned value is anywhere from d to infinity
def depth(r, d=1):
	while (r > random.random()):
		d += 1
	return d

# Used to change frequencies of patterns and bases. Mostly used for videos with many patterns to have subtle changes in what kinds of patterns emerge. 
def changer(l, minimum):
	a = int(random.random() * len(l))

	while l[a][1] <= minimum:
		a = int(random.random() * len(l))

	l[a][1] -= 1
	l[int(random.random() * len(l))][1] += 1

	return l

# Helper function used to turn rgb(r, g, b) into moviepy colors
def moviepyColor(colorName):
	rgb_values = colorName.replace("rgb(", "").replace(")", "").split(",")
	return tuple(map(int, rgb_values))

# Applies two patterns one every frame, an "up" pattern and a "down" pattern.
# The "up" pattern is increasing in visibility while "down" decreases in visibility.
# This is so that we can apply multiple patterns over a whole video so the output is not slow or boring with a single pattern and still have gradual change
def effect(get_frame, t):
	global height, width, thePattern, duration

	# How far along in the animation should each pattern be
	upCount = (t % halfstep) / step
	downCount = (upCount + 0.5) % 1

	# Getting linear spaces for efficient pattern application
	x = np.linspace(0, 1, width, dtype=np.float32)
	y = np.linspace(0, 1, height, dtype=np.float32)
	xx, yy = np.meshgrid(x, y)
	ttDown = np.full_like(xx, downCount)
	ttUp = np.full_like(xx, upCount)

	# Percentage opacity (percent is for "up", 1 - percent is for "down")
	a = len(thePattern) - 1 - int(t / halfstep)
	percent = (t % halfstep) / halfstep

	# print("downCount: " + str(downCount))
	# print("upCount: " + str(upCount))
	# print("a: " + str(a))
	# print("percent: " + str(percent))

	# Get the frame colors for the two patterns
	frameDown = (255 * thePattern[a].apply(xx, yy, ttDown)).astype(np.uint8).transpose(1, 2, 0)
	frameUp   = (255 * thePattern[a - 1].apply(xx, yy, ttUp)).astype(np.uint8).transpose(1, 2, 0)

	# Combine them based on opacity percents
	frame = frameDown * (1 - percent) + frameUp * percent

	# print("downCount: " + str(downCount) + ", aDown: " + str(a) + ", %: " + str(1 - percent))
	# print("upCount: " + str(upCount) + ", aUp: " + str(a - 1) + ", %: " + str(percent))

	# Post-processing
	frame[..., 0] = np.clip(frame[..., 0] * 1.2, 0, 255)  # boost red
	frame[..., 1] = np.clip(frame[..., 1] * 0.9, 0, 255)  # soften green
	frame[..., 2] = np.clip(frame[..., 2] * 0.6, 0, 255)  # reduce blue

	return frame

# Generate the video with the effect
def generate_with_effect():
	global height, width, duration, fps

	background_clip = ColorClip(size=(width, height), color=moviepyColor("rgb(0,0,0)"), duration=duration)
	effected_clip = background_clip.transform(effect)
	# effected_clip = background_clip

	effected_clip.write_videofile("background.mp4", fps=fps)

if __name__ == '__main__':
	random.seed(seed)

	# Get the patterns used in the video
	thePattern = []
	for i in range(patterns):
		# Get the depth
		# d = depth(0.5, d=2)
		# print(d)
		d = 3

		# Create and add the pattern
		thePattern.append(getPattern(d))

		# Change frequencies
		pattern = changer(pattern, 0)
		baseInfo = changer(baseInfo, 1)

	generate_with_effect()