
import sys
import numpy as np
from perlin_noise import PerlinNoise



def gen(size, dimension=(256, 256), octaves=1, output_path='./temp/perlin-%d.npy'):
	arr = np.zeros((size, dimension[0], dimension[1]), np.float32)

	for i in range(size):
		noise = PerlinNoise(octaves = octaves, seed = i + 100)

		for y in range(dimension[0]):
			for x in range(dimension[1]):
				arr[i, y, x] = noise([y / dimension[0], x / dimension[1]])

		var = np.var(arr[i])
		arr[i] /= var

	with open(output_path % octaves, 'wb') as file:
		np.save(file, arr)


if __name__ == '__main__':
	# gen(int(sys.argv[1]), octaves = int(sys.argv[2]))
	gen(size = 5, octaves=10)
