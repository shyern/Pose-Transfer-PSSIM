import os
import shutil
from PIL import Image

IMG_EXTENSIONS = [
'.jpg', '.JPG', '.jpeg', '.JPEG',
'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	new_root = '/home/haoyue/codes/datasets/fashion_data/'
	if not os.path.exists(new_root):
		os.mkdir(new_root)

	train_root = '/home/haoyue/codes/datasets/fashion_data/train/'
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = '/home/haoyue/codes/datasets/fashion_data/test/'
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open('./fashion_data/train.lst', 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open('./fashion_data/test.lst', 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	print(train_images, test_images)
	

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				path_names = path.split('/') 
				# path_names[2] = path_names[2].replace('_', '')
				# path_names[3] = path_names[3].replace('_', '')
				# path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
				# path_names = "".join(path_names)
				new_path = 'fashion'+path_names[7]+path_names[8]+path_names[9].replace('_','')+path_names[10].split('_')[0] + "_" + "".join(path_names[10].split('_')[1:])
				# new_path = os.path.join(root, path_names)
				img = Image.open(path)
				imgcrop = img.crop((40, 0, 216, 256))
				if new_path in train_images:
					imgcrop.save(os.path.join(train_root, new_path))
				elif new_path in test_images:
					imgcrop.save(os.path.join(test_root, new_path))

make_dataset(r'/home/haoyue/codes/datasets/In-shop Clothes Retrieval Benchmark/img/')