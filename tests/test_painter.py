import sys
import os
sys.path.insert(0, os.path.abspath('../cvjson/'))
sys.path.insert(0, os.path.abspath('../extensions/'))
#print(sys.path)

import unittest
from cvjson.cvj import CVJ
from cvjson.extensions.painter import Painter
import cv2
import shutil
import pickle
from multiprocessing import Manager
import json

json_path = "/home/typhon/Downloads/full_xview_coco_.json" #TODO make a testing only json
image_folder_path = "/home/typhon/Downloads/full_view_test" #TODO make a testing only image folder Make 2. padded and unpadded

image_save_dir = './save_dir/'

class TestPainter(unittest.TestCase):

    def test_generate_negatives(self):

        cvj_obj = CVJ(json_path=json_path)
        cvj_obj.image_folder_path = image_folder_path


        painter = Painter(cvj_obj)

        # Checkpoints are no longer being tested.
        filepaths = painter.generate_negatives(image_save_dir, 2, 0, generation_method=Painter.INPAINT)

        
        for file in filepaths:
            self.assertIn(os.path.basename(file), list(painter.get_filename_2_image_id().keys()))

        remove_list = painter.remove_generated(Painter.ALL)

        for img in painter["images"]:
            for file in remove_list:
                self.assertNotEqual(file, img["file_name"])
        