''' 
    annotations  :
                [{
                    "id": int,
                    "image_id": int,
                    "category_id": int,
                    "segmentation": RLE or [polygon],
                    "area": float,
                    "bbox": [x,y,width,height],
                    "iscrowd": 0 or 1,
                    }]

    categories  : [{
                    "id": int,
                    "name": str,
                    "supercategory": str,
                     }],

    images      :  [{
                    "id": int,
                    "file_name": str,
                    "width": int,
                    "height": int
                    }],

More information on how this structure is chosen read "Introduction to the CVJ"
'''

import sys
import os
sys.path.insert(0, os.path.abspath('../cvjson/'))
sys.path.insert(0, os.path.abspath('../extensions/'))
#print(sys.path)

import unittest
from cvjson.cvj import CVJ
from cvjson.extensions.cropper import Cropper
import cv2
import shutil
  

json_path = "/home/typhon/Downloads/resplit_002_trn.json" #TODO make a testing only json
image_folder_path = "/home/typhon/Desktop/workspace/inpainted_with_regular_image" #TODO make a testing only image folder Make 2. padded and unpadded

image_save_dir = './save_dir/'

class TestCropper(unittest.TestCase):
    

    
    def test_crop_images_bbox_centered(self):

        cvj_obj = CVJ(json_path)
        cvj_crop = Cropper(cvj_obj, image_folder_path)


        max_img_size = 768
        min_img_size = 128
        max_bounding_box = 600
        scal = 1
        pad = 3000
        threshold = 10
        enum = cv2.BORDER_REFLECT
        test = 2



        #TODO make this work with padding of 0
        timestamp = cvj_crop.crop_images_bbox_centered(image_save_dir, max_image_size = max_img_size , min_image_size = min_img_size,
                                  max_bounding_box_size=max_bounding_box, scale = scal, padding = pad, image_threshold=threshold, cv2enum=enum, testing=test)

        train_file = str(timestamp) + "_coco_train.json"
        train_file = os.path.join(image_save_dir, train_file)

        cvj_obj = CVJ(train_file)

        self.assertIn("images", list(cvj_obj.keys()))
        self.assertIn("categories" ,list(cvj_obj.keys()))
        self.assertIn("annotations", list(cvj_obj.keys()))

        for ann in cvj_obj["annotations"]:
            self.assertIn("id", ann)
            self.assertIn("image_id", ann)
            self.assertIn("category_id", ann)
            self.assertIn("segmentation", ann)
            self.assertIn("area", ann)
            self.assertIn("bbox", ann)
            self.assertIn("iscrowd", ann)

        for img in cvj_obj["images"]:
            self.assertIn("id", img)
            self.assertIn("file_name", img)
            self.assertIn("width", img)
            self.assertIn("height", img)

        for cat in cvj_obj["categories"]:
            self.assertIn("id", cat)
            self.assertIn("name", cat)
            self.assertIn("supercategory", cat)


        train_images = str(timestamp) + "_trn"
        train_images = os.path.join(image_save_dir, train_images)

        cvj_obj.image_folder_path = train_images

        for img_id, filepath in cvj_obj.get_image_id_2_filepath().items():
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

            height, width = img.shape[:2]

            self.assertGreaterEqual(height, min_img_size)
            self.assertGreaterEqual(width, min_img_size)

            self.assertLessEqual(height, max_img_size)
            self.assertLessEqual(width, max_img_size)

        if threshold > 0:
            catlist =[]
            amount_of_ids = len(cvj_obj["categories"])
            for cat in cvj_obj["categories"]:
                catlist.append(int(cat["id"]))

            print(sorted(catlist))

            amount_of_images_supposed_to_be_created_and_stratified = amount_of_ids * threshold

            actual_amount_of_images_in_json = len(cvj_obj["images"])

            self.assertEqual(actual_amount_of_images_in_json, amount_of_images_supposed_to_be_created_and_stratified)


        # Removes the files that it was testing.  This is just tidying up the directory
        shutil.rmtree(image_save_dir)

    #TODO Come up with tests for these files.
    #     strat_file = str(timestamp) + "_stratified_images.json"
    #     class_count_file = str(timestamp) + "_image_class_counts.json"
    #     augment_file = str(timestamp) + "_augmented_image_class_filepaths.json"
    #     too_big_file = str(timget_image_id_2_filepath"



    