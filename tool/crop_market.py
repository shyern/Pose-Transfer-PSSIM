from PIL import Image
import os

img_dir = './results/results3/market_PATN_ganssimperl1_win7_batch32/test_660/images'
save_dir = './results/results3/market_PATN_ganssimperl1_win7_batch32/test_660/images_crop'
# market_PATN_ganmspartssimperl1l1
# market_PATN_ganmspartssim2

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

cnt = 0
for item in os.listdir(img_dir):
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	cnt = cnt + 1
	print('%d/12000 ...' %(cnt))
	img = Image.open(os.path.join(img_dir, item))
	# for 5 split
	imgcrop = img.crop((256, 0, 320, 128))
	imgcrop.save(os.path.join(save_dir, item))
