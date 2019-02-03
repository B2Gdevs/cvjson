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
from tqdm import tqdm

  

json_path = "full_xview_coco_.json" #TODO make a testing only json
image_folder_path = "images" #TODO make a testing only image folder Make 2. padded and unpadded
#/home/ben/Desktop/CV_JSON/cvjson/tests/test_cvj.py
image_save_dir = './save_dir/'

cvj_obj = CVJ(json_path=json_path)
cvj_obj.image_folder_path = image_folder_path

test_json_data = cvj_obj.load_json(json_path)

class TestCVJ(unittest.TestCase):

    def test_get_dictionary(self):
        """ 
        Enums
        -----
        Dictionary enums:
                        * IMID_2_ANNS , Image ID to Annoations
                        * CLID_2_NAME, Class ID to Class Name
                        * CLNAME_2_CLID, Class name to Class ID
                        * IMID_2_FNAME, Image ID to File Name
                        * FNAME_2_IMID, File Name to Image ID
                        * IMID_2_FPATH, Image ID to File Path
                        * IMID_2_IMATTR, Image Id to Image Attributes
                        * CLID_2_ANNS, Class ID to Annotations
            
        """

        image_id_2_anns = cvj_obj.get_dictionary(CVJ.IMID_2_ANNS)
        class_id_2_class_name = cvj_obj.get_dictionary(CVJ.CLID_2_NAME)
        class_name_2_class_id = cvj_obj.get_dictionary(CVJ.CLNAME_2_CLID)
        image_id_2_file_name = cvj_obj.get_dictionary(CVJ.IMID_2_FNAME)
        file_name_2_image_id = cvj_obj.get_dictionary(CVJ.FNAME_2_IMID)
        image_id_2_file_path = cvj_obj.get_dictionary(CVJ.IMID_2_FPATH)
        image_id_2_image_attributes = cvj_obj.get_dictionary(CVJ.IMID_2_IMATTR)
        class_id_2_annotations = cvj_obj.get_dictionary(CVJ.CLID_2_ANNS)

        self.assertEqual(image_id_2_anns, cvj_obj.get_image_id_2_anns())
        self.assertEqual(class_id_2_class_name, cvj_obj.get_class_id_2_name())
        self.assertEqual(class_name_2_class_id, cvj_obj.get_class_name_2_id())
        self.assertEqual(image_id_2_file_name, cvj_obj.get_image_id_2_filename()) 
        self.assertEqual(file_name_2_image_id, cvj_obj.get_filename_2_image_id())
        self.assertEqual(image_id_2_file_path, cvj_obj.get_image_id_2_filepath())
        self.assertEqual(image_id_2_image_attributes, cvj_obj.get_image_id_2_image_attribs())
        self.assertEqual(class_id_2_annotations, cvj_obj.get_class_id_2_anns())

    def test_get_image_id_2_image_attribs(self):
        
        img_id = 50
        img_attr = cvj_obj.get_image_id_2_image_attribs(img_id)

        img_id_dict = cvj_obj.get_image_id_2_image_attribs()

        attrib = None

        for img in tqdm(cvj_obj["images"]):
            if img["id"] == img_id:
                attrib = img

        self.assertEqual(img_attr, attrib)

        self.assertEqual(img_attr, img_id_dict[img_id])

        test_dict = cvj_obj.get_image_id_2_image_attribs(json_data=test_json_data)

        self.assertEqual(test_dict, img_id_dict)

    def test_get_image_id_2_filepath(self):

        img_id = 11
        image_name = '1886.tif'
        actual_path = os.path.join(image_folder_path, image_name)

        cvj_obj.image_folder_path = image_folder_path
        img_file_path = cvj_obj.get_image_id_2_filepath(img_id)
        img_id_dict = cvj_obj.get_image_id_2_filepath()

        self.assertEqual(actual_path, img_file_path)

        self.assertEqual(actual_path, img_id_dict[img_id])

    
    def test_get_filename_2_image_id(self):

        image_name = '1886.tif'
        img_id = cvj_obj.get_filename_2_image_id(image_name)

        names_dict = cvj_obj.get_filename_2_image_id()

        id = None

        for img in tqdm(cvj_obj["images"]):
            if img["file_name"] == image_name:
                id = img["id"]

        self.assertEqual(id, img_id)

        self.assertEqual(id, names_dict[image_name])

        test_dict = cvj_obj.get_filename_2_image_id(json_data=test_json_data)

        self.assertEqual(test_dict, names_dict)

    def test_get_image_id_2_filename(self):

        img_id = 50
        file_name = cvj_obj.get_image_id_2_filename(img_id)

        img_id_dict = cvj_obj.get_image_id_2_filename()

        name = None

        for img in tqdm(cvj_obj["images"]):
            if img["id"] == img_id:
                name = img["file_name"]

        self.assertEqual(file_name, name)

        self.assertEqual(file_name, img_id_dict[img_id])

        test_dict = cvj_obj.get_image_id_2_filename(json_data=test_json_data)

        self.assertEqual(test_dict, img_id_dict)

    def test_get_class_name_2_id(self):

        class_name = '50'
        id = cvj_obj.get_class_name_2_id(class_name)

        names_dict = cvj_obj.get_class_name_2_id()

        id_for_50 = None

        for cat in tqdm(cvj_obj["categories"]):
            if cat["name"] == class_name:
                id_for_50 = cat["id"]

        self.assertEqual(id, id_for_50)

        self.assertEqual(id, names_dict[class_name])


        test_dict = cvj_obj.get_class_name_2_id(json_data=test_json_data)

        self.assertEqual(test_dict, names_dict)

    def test_get_class_id_2_name(self):

        class_id = 50
        name = cvj_obj.get_class_id_2_name(class_id)

        names_dict = cvj_obj.get_class_id_2_name()

        name_for_50 = None

        for cat in tqdm(cvj_obj["categories"]):
            if cat["id"] == class_id:
                name_for_50 = cat["name"]

        self.assertEqual(name, name_for_50)

        self.assertEqual(name, names_dict[class_id])


        test_dict = cvj_obj.get_class_id_2_name(json_data=test_json_data)

        self.assertEqual(test_dict, names_dict)

    def test_get_image_id_2_anns(self):

        img_id = 1
        anns = cvj_obj.get_image_id_2_anns(img_id)
        anns_dict = cvj_obj.get_image_id_2_anns()
        test_anns_dict  = cvj_obj.get_image_id_2_anns(json_data=test_json_data)

        ann_for_one = []

        for ann in tqdm(cvj_obj["annotations"]):
            if ann["image_id"] == img_id:
                ann_for_one.append(ann)

        for ann in anns:
            self.assertIn(ann, ann_for_one)

        self.assertEqual(ann_for_one, anns_dict[img_id])

        self.assertEqual(test_anns_dict, anns_dict)


    def test_remove_list(self):

        remove_list = cvj_obj.remove_by_name(["1886.tif"], save=False)

        for img in tqdm(cvj_obj["images"]):
            for file in remove_list:
                self.assertNotEqual(file, img["file_name"])

    def test_get_category_ids(self):

        ids = cvj_obj.get_category_ids()
        actual_ids = []

        NU = [actual_ids.append(cat["id"]) for cat in cvj_obj["categories"]]
        for id in tqdm(ids):
            self.assertIn(id, actual_ids )

    def test_get_category_names(self):

        names = cvj_obj.get_category_names()
        actual_names = []

        NU = [actual_names.append(cat["name"]) for cat in cvj_obj["categories"]]
        for name in names:
            self.assertIn(name, actual_names )

    '''
    def test_get_annotations(self):

        anns = cvj_obj.get_annotations()

        actual_anns = [ann for ann in cvj_obj["annotations"]]

        for ann in tqdm(anns):

            self.assertIn(ann, actual_anns )
    '''

    def test_get_filenames(self):

        names = cvj_obj.get_filenames()

        actual_names = [img["file_name"] for img in cvj_obj["images"]]

        for name in tqdm(names):

            self.assertIn(name, actual_names )

    
