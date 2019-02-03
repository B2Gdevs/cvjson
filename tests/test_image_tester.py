import sys
import os
sys.path.insert(0, os.path.abspath('../cvjson/'))
sys.path.insert(0, os.path.abspath('../extensions/'))
#print(sys.path)

import unittest
from cvjson.cvj import CVJ
from cvjson.extensions. image_tester import Image_Tester
import cv2
import shutil
  

json_path = "/home/typhon/Downloads/resplit_002_trn.json" #TODO make a testing only json
image_folder_path = "/home/typhon/Desktop/workspace/dlib_xview/dlib-xview/xview_no_dlib/cvjson/unit_tests/dead_images_img_tester" #TODO make a testing only image folder Make 2. padded and unpadded

image_save_dir = './save_dir/'

class TestImage_Tester(unittest.TestCase):

    def test_remove_unreadable_files(self):

        cvj_obj = CVJ(json_path=json_path)
        cvj_obj.image_folder_path = image_folder_path
        tester = Image_Tester(cvj_obj)

        image_names = cvj_obj.get_filename_2_image_id().keys()

        for file in os.listdir(image_folder_path):
            self.assertIn(file, image_names)

        tester.remove_unreadable_files()

        for img in tester["images"]:
            self.assertNotEqual("8.png", img["file_name"])