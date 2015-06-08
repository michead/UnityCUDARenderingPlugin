from sys import argv

FILENAME = argv[1]
SIDE_SIZE = int(argv[2])
SIDE_RANGE = float(argv[3])


def compute_mesh():
	global FILENAME, SIDE_SIZE, SIDE_RANGE

	with open(FILENAME, 'w') as f:

		# COMPUTING VERTICES
		for i in xrange(SIDE_SIZE ** 2):
			h = i / SIDE_SIZE
			v = i % SIDE_SIZE

			x = ((2.* SIDE_RANGE) / SIDE_SIZE) * h - SIDE_RANGE
			z = ((2.* SIDE_RANGE) / SIDE_SIZE) * v - SIDE_RANGE

			f.write('v %f %f %f\n' % (x, 0., z))

		# COMPUTING FACES
		for i in xrange(SIDE_SIZE ** 2):
			if((i + 1) % SIDE_SIZE != 0):
				if (i + SIDE_SIZE < SIDE_SIZE ** 2):
					f.write('f %d %d %d\n' % (i + 1 + 1, i + SIDE_SIZE + 1, i + 1))
				if(i - SIDE_SIZE > -1):
					f.write('f %d %d %d\n' % (i + 1 + 1, i + 1, i - SIDE_SIZE + 1 + 1))


compute_mesh()