
import os
import cv2
import math
import numpy as np



def randint (low, high):
	return np.random.randint(low, high) if high > low else 0


def logrand (low, high):
	ln_low = np.log(low)
	ln_high = np.log(high)

	return np.exp(ln_low + np.random.random() * (ln_high - ln_low))


class TextureSetIterator:
	def __init__ (self, frames):
		self.frames = frames
		self.iterator = iter(self)

	def __iter__ (self):
		while True:
			for frame in self.frames:
				yield frame

	def __next__ (self):
		return next(self.iterator)

	def get (self, size, scale_range = (0.5, 2), blur_range = (0, 5)):
		frame = next(self)

		scale = logrand(scale_range[0], scale_range[1])

		crop_size = (math.floor(size[0] * scale), math.floor(size[1] * scale))
		tiling = (math.ceil(crop_size[0] / frame.shape[0]), math.ceil(crop_size[1] / frame.shape[1]), 1)
		if tiling[0] <= 1 and tiling[1] <= 1:
			left = randint(0, frame.shape[1] - crop_size[1])
			top = randint(0, frame.shape[0] - crop_size[0])
			image = frame[top:top + crop_size[0], left:left + crop_size[1], :]
			image = cv2.resize(image, size[::-1])
		else:
			tiling_frame = cv2.resize(frame, (math.ceil(frame.shape[1] / scale), math.ceil(frame.shape[0] / scale)))
			if len(tiling_frame.shape) < 3:
				tiling_frame = np.expand_dims(tiling_frame, -1)
			tiling_frame = np.tile(tiling_frame, tiling)

			left = randint(0, tiling_frame.shape[1] - size[1])
			top = randint(0, tiling_frame.shape[0] - size[0])
			image = tiling_frame[top:top + size[0], left:left + size[1], :]

		blur_kernel = randint(blur_range[0], blur_range[1]) * 2 + 1
		if blur_kernel > 1:
			image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

		if len(image.shape) < 3:
			image = np.expand_dims(image, -1)

		return image


def loadOneImage (names):
	while True:
		filename = next(names)
		if filename is None:
			return None

		image = cv2.imread(filename)
		if image is not None:
			return image

		print('image loading failed:', filename)


class TextureSet:
	def __init__ (self, dir):
		self.dir = dir
		self.filenames = os.listdir(dir)

	def makeIterator (self, unit_size = 256, frame_size = (2048, 2048), frame_count = 16, mono = True, shuffle = True):
		print('Making TextureSet iterator...')
		uw, uh = (frame_size[0] // unit_size, frame_size[1] // unit_size)
		required_unit_count = uw * uh * frame_count

		if shuffle:
			np.random.shuffle(self.filenames)

		names = self.filenames
		if required_unit_count > len(names):
			names *= (required_unit_count // len(names) + 1)
		names = map(lambda name: os.path.join(self.dir, name), names)
		#print('names:', names)

		frames = []
		for f in range(frame_count):
			frame = np.zeros(frame_size + (1 if mono else 3,), dtype=np.uint8)
			for y in range(uh):
				for x in range(uw):
					image = loadOneImage(names)
					if image is None:
						raise ValueError(f'TextureSet folder file count is not sufficient: {len(frames)}/{frame_count}')

					if mono:
						image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

					h, w, *_ = image.shape
					if h < unit_size or w < unit_size:
						scale = unit_size / min(h, w)
						image = cv2.resize(image, (round(w * scale), round(h * scale)))
						h, w, *_ = image.shape
					left = randint(0, w - unit_size) if shuffle else (w - unit_size) // 2
					top = randint(0, h - unit_size) if shuffle else (h - unit_size) // 2
					if mono:
						image = np.expand_dims(image, -1)
					image = image[top:top + unit_size, left:left + unit_size, :]

					frame[y * unit_size:(y + 1) * unit_size, x * unit_size:(x + 1) * unit_size, :] = image

			frames.append(frame)

		print('TextureSet iterator done.')

		return TextureSetIterator(frames)
