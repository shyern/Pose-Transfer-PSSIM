from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='/home/haoyue/codes/results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
        # self.parser.add_argument("--generated_images_dir",
        #                     default='/data0/haoyue/codes/results/market_PATN_ssim/test_latest/images',
        #                     help='Folder with images')
        # self.parser.add_argument("--annotations_file_test",
        #                     default='/data0/haoyue/codes/datasets/market_data/market-annotation-test.csv', help='')
        # self.parser.add_argument("--bodypart_mask_dir", default='/data0/haoyue/codes/datasets/market_data/testM/', help='')

        self.isTrain = False
