from PIL import Image
import os
import cv2

# #读入原始图像
# img=cv2.imread('figs.png')
# #灰度化处理
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# cv2.imwrite('figs2.png', gray)


# img_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_pSSIM\fashion'
# img_dir = r'E:\codes\present_codes\my_codes\Person-Image-Gen\records\results_fig\market'
img_dir = r'E:\codes\present_codes\my_codes\Person-Image-Gen\records\results_fig\fashion'
method = 'def-gan'
img_dir = os.path.join(img_dir,method)

# # img_loc = (0,0,64,128) # source image
# img_loc = (0,0,176,256)
# save_dir = os.path.join(img_dir, 'source_image')
# # save_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_PATN\fashion/source_image'

# # img_loc = (64,0,128,128)
# img_loc = (176,0,176*2,256)
# save_dir = os.path.join(img_dir, 'source_pose')
# # save_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_PATN\fashion/source_pose'

# # img_loc = (128,0,192,128)
# img_loc = (176*2,0,176*3,256)
# save_dir = os.path.join(img_dir, 'target_image')
# # save_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_PATN\fashion/target_image'


# # img_loc = (192,0,256,128)
# img_loc = (176*3,0,176*4,256)
# save_dir = os.path.join(img_dir, 'target_pose')
# # save_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_PATN\fashion/target_pose'

# img_loc = (256,0,320,128)
# img_loc = (128,0,192,128)
# img_loc = (176*4,0,176*5,256)
img_loc = (256*2+40,0,256*3-40,256)
save_dir = os.path.join(img_dir, 'generated_image')
# save_dir = r'E:\codes\present codes\Pose-Transfer-master\results\failure_pSSIM\fashion/generated_image'



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
	imgcrop = img.crop(img_loc)
	imgcrop.save(os.path.join(save_dir, item))
