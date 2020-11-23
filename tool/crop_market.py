from PIL import Image
import os
import pandas as pd
from shutil import copyfile



# img_dir = '/data0/haoyue/codes/results_v2/market_PATN_cat/test_latest/images'
# save_dir_generated = '/data0/haoyue/codes/results_v2/market_PATN_cat/test_latest/images_generated'
# save_dir_gt = '/data0/haoyue/codes/results_v2/market_PATN_cat/test_latest/images_gt'

# market_PATN_ganmspartssimperl1l1
# market_PATN_ganmspartssim2

# if not os.path.exists(save_dir_generated):
# 	os.mkdir(save_dir_generated)
#
# if not os.path.exists(save_dir_gt):
# 	os.mkdir(save_dir_gt)

# cnt = 0
# for item in os.listdir(img_dir):
# 	if not item.endswith('.jpg') and not item.endswith('.png'):
# 		continue
# 	cnt = cnt + 1
# 	print('%d/12000 ...' %(cnt))
# 	img = Image.open(os.path.join(img_dir, item))
# 	# for 5 split
# 	imgcrop = img.crop((256, 0, 320, 128))
# 	imggt = img.crop((128, 0, 192, 128))
# 	imgcrop.save(os.path.join(save_dir_generated, item))
# 	imggt.save(os.path.join(save_dir_gt, item))


def copy_file(annotations_file, source_path, target_path):
	pairs_file_train = pd.read_csv(annotations_file)
	size = len(pairs_file_train)
	print('Loading data pairs ...')
	for i in range(size):
		name = pairs_file_train.iloc[i]['to']
		copyfile(source_path+'/'+name, target_path+'/'+name)
		print(name)


	# annotations_file = pd.read_csv(annotations_file, sep=':')
	# annotations_file = annotations_file.set_index('name')
	# # image_size = (128, 64)
	# # # image_size = (256, 176)
	# cnt = len(annotations_file)
	# for i in range(cnt):
	# 	print('processing %d / %d ...' % (i, cnt))
	# 	row = annotations_file.iloc[i]
	# 	name = row.name
	# 	copyfile(source_path+'/'+name, target_path+'/'+name)
	# 	print(name)

if __name__== "__main__":
	annotations_file = '/data0/haoyue/codes/datasets/market_data/market-pairs-train.csv'
	source_path = '/data0/haoyue/codes/datasets/market_data/test'
	target_path = '/data0/haoyue/codes/datasets/market_data/image_gt'

	if not os.path.exists(target_path):
		os.mkdir(target_path)

	copy_file(annotations_file, source_path, target_path)