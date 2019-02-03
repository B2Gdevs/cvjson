 

import os
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from ..cvj import CVJ      
from .visualizer import Visualizer
from itertools import repeat
import subprocess
import shutil
import pickle
import pandas as pd
from tqdm import tqdm

import re
from decimal import Decimal

from xml.etree.ElementTree import Element, SubElement, Comment
import xml.etree.ElementTree as ET
from xml.dom import minidom

 

class Data_Loader(CVJ):
    """
    The Data_Loader class is used to load any
    data that is not in COCO JSON format and is 
    supported by this library.

    Currently the CVJ library supports the following
    external data formats:
    
    Formats
    -------

    NeoVision

    """

    def __init__(self, cvj): 
        super().__init__()


        #### Super Class Vars Start
        self._json_path = cvj._json_path
        self._json_data = cvj._json_data  
        self._image_folder_path = cvj._image_folder_path
        self._image_class_counts_path = cvj._image_class_counts_path
        self._class_to_filepath_data = cvj._class_to_filepath_data
        self._image_id_2_anns_dict = cvj._image_id_2_anns_dict
        self._class_id_2_name_dict = cvj._class_id_2_name_dict
        self._class_name_2_id_dict = cvj._class_name_2_id_dict
        self._img_id_2_filename_dict = cvj._img_id_2_filename_dict
        self._filename_2_image_id_dict = cvj._filename_2_image_id_dict
        self._imageid_to_filepath_dict = cvj._imageid_to_filepath_dict
        self._image_id_2_image_attribs = cvj._image_id_2_image_attribs
        self._class_id_2_anns_dict = cvj._class_id_2_anns_dict
        ### Super Class Vars End

    def _is_similar(self, image1, image2):
        return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())


    def load_neovision(self, neovision_path, path_to_video, generate_image_folder=False, visualize=False):
        
        cat_name_to_id = {} # to house the category name to the arbitrarily made category ID
        cat_id = 0

        new_json = CVJ().create_empty_json()
        if generate_image_folder:
            path_name = os.path.join(os.path.dirname(path_to_video[0]), "neo_training_images")
            subprocess.call(['mkdir', '-p', path_name])

        # Just note that the dataset has labeled a crowd of planes, except that they aren't annotated as
        # a crowd.  Therefore neovision data could be problematic for training any reliable model.

        path_and_video_index = 0
        ann_id = 0
        image_count = 0
        for path in neovision_path:
            df = pd.read_csv(path)

            # X3 and Y3 are the bottom right corner, X1 and Y1 are top left corner
            df = df.query("BoundingBox_X1 < BoundingBox_X3")
            df = df.query("BoundingBox_Y1 < BoundingBox_Y3")

            frames = df.Frame.unique()

            video_name = os.path.basename(path_to_video[path_and_video_index]).split(".")[0]

            video = cv2.VideoCapture(path_to_video[path_and_video_index])

            frame_to_image = {}
            frame_counter = 0
            print("\n Reading frames of video.  Please wait.")
            while True:
                success, img = video.read()
                if success:
                    frame_to_image[frame_counter] = img
                    frame_counter += 1
                else:
                    break

            img_name_to_img_id = {}
            frame_to_image_name = {}
            for frame in tqdm(frames, desc="Inserting images from video"):

                img_name = video_name + "_" + str(frame) + ".png"
                img = frame_to_image[int(frame)]
                new_json["images"].append(self.entry_img(img_name, img.shape[0], img.shape[1], image_count))
                img_name_to_img_id[img_name] = image_count
                frame_to_image_name[int(frame)] = img_name
                image_count += 1

                if generate_image_folder:
                    img_name_path = os.path.join(path_name, img_name)
                    cv2.imwrite(img_name_path, img)


            categories = df.ObjectType.unique()
            categories = [x for x in categories if str(x) != "nan"]

            for cat in tqdm(categories, desc="Finding categories and inserting categories"):
                if cat not in cat_name_to_id:
                    cat_name_to_id[cat] = cat_id
                    new_json["categories"].append(self.new_category(int(cat_id), cat))

                    cat_id += 1

            
            for index, row in tqdm(df.iterrows(), desc="Creating and inserting annotations"):
                """
                [{
                    "id": int,
                    "image_id": int,
                    "category_id": int,
                    "segmentation": RLE or [polygon],
                    "area": float,
                    "bbox": [x,y,width,height],
                    "iscrowd": 0 or 1,
                    }]
                """
                ann = {}

                x0 = row.BoundingBox_X1
                y0 = row.BoundingBox_Y1
                w = row.BoundingBox_X3 - row.BoundingBox_X1 
                h = row.BoundingBox_Y3 - row.BoundingBox_Y1

                do_next_error = True # this is for the try catch statements.  I dont want a specific one called if another is.

                try:
                    ann = self.entry_bbox([x0,y0,w,h], cat_name_to_id[row.ObjectType], img_name_to_img_id[frame_to_image_name[int(row.Frame)]], ann_id )
                    new_json["annotations"].append(ann)
                    ann_id += 1                

                except KeyError as e:
                    try:
                        cat_name_to_id[row.ObjectType]
                    except KeyError as e:
                        print("A Key Error was found with the ObjectType {} in row {} from file {}".format(row.ObjectType,index + 1, os.path.basename(path)))
                        print("Row = {}".format(row))
                        print("\nStill continuing to process files.")
                        continue
                    try:
                        img_name_to_img_id[frame_to_image_name[int(row.Frame)]]
                    except KeyError as e:
                        try:
                            frame_to_image_name[int(row.Frame)]
                        except KeyError as e:
                            print("The Frame number at row {} in the file {} could not be found in the frame of the video {}".format(index + 1, path, os.path.basename(path_to_video[path_and_video_index])))
                            print("Row = {}".format(row))
                            do_next_error = False
                        if do_next_error:
                            print("A key error occurred.  The program could not find an image name to image id.  This is only possible if the number of frames\n"\
                                "the video do not match the number of frames that the CSV file is saying the video has.\n" \
                                "Video = {}.\nFrame Count = {}\nCSV file = {}\nLargest Frame Number in CSV = {}".format(os.path.basename(path_to_video[path_and_video_index]),\
                                len(frames), path, df.Frame.max()))
                        do_next_error = True

            frame_to_image.clear()

            path_and_video_index += 1

        try:
            self.clear()
        except:
            print("No internal data found.  Setting data from new file.")

        self._json_data = new_json

        print("Neovision data successfully loaded in to this current instance. \n Please remember when saving"\
                " to include a save path.  Using the method save_internal_json(save_name= your/path.json) is recommended")
        
        if generate_image_folder:
            print("Images successfully created at {}".format(path))

        if visualize:
            self.image_folder_path = path_name
            Visualizer(self).visualize_bboxes()


        return self

    def load_mardct(self, path_to_mardct, path_to_video, visualize=False):

        # The bounding boxes still have problems.  They lose the ships after a few frames.
        
        new_json = self.create_empty_json()

        # I need to make each frame unique if collecting videos.
        video_name = os.path.basename(path_to_video)

        video = cv2.VideoCapture(path_to_video)

        frame_to_image = {}
        frame_counter = 0
        print("\n Reading frames of video.  Please wait.")
        while True:
            success, img = video.read()
            if success:
                frame_to_image[frame_counter] = img
                frame_counter += 1
            else:
                break

        # Mardct data from tracking comes in two forms.  One being xml
        # and the other being a format I do not know.  First we will check for
        # xml and second we will check for the other format type.

        is_xml = True
        try:
            root = ET.parse(path_to_mardct).getroot()
            frames = root.findall("frame")
        except:
            is_xml = False
            print("XML file not detected, trying other format")

        if is_xml:

            #################################### XML FILE PARSING ####################################

            list_of_categories =[]
            img_id_counter = 0
            ann_id_counter = 0
            for frame in frames:
                frame_number = int(frame.attrib["number"])
                print(frame_number)

                # used for setting images
                image_name = video_name + str(frame_number)
                img = frame_to_image[frame_number]
                new_json["images"].append(self.entry_img(image_name, img.shape[0], img.shape[1],img_id_counter))

                img_id_counter += 1
                
                obj_list_element = frame.find("objectlist")
                obj_elements = obj_list_element.findall("object")
                for object_ in obj_elements:

                    if int(object_.attrib["id"]) not in list_of_categories:
                        list_of_categories.append(int(object_.attrib["id"]))
                    box = object_.find("box")

                    #xc and yc are center coordinates
                    # the coordinates for xc, and yc were floating points, but they need to be integers.  Therefore
                    # the bounding boxes may be getting shifted away from the ships because of that.
                    x0 = int(float(box.attrib["xc"])) - int(float(box.attrib["w"]))
                    y0 = int(float(box.attrib["yc"])) - int(float(box.attrib["h"]))
                    w = 2 * int(float(box.attrib["w"]))
                    h = 2 * int(float(box.attrib["h"]))

                    new_json["annotations"].append(self.entry_bbox([x0, y0, w, h], int(object_.attrib["id"]), img_id_counter, ann_id_counter ))
                    ann_id_counter += 1

                    ########################## Visualizing #################################
                    if visualize:
                        x1 = x0 + w
                        y1 = y0 + h
                        cv2.rectangle(img, (x0,y0), (x1,y1), (255, 60, 36), thickness= 2)

                if visualize:
                    cv2.imshow("test",img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    ########################## Visualizing #################################



                if img_id_counter > 2:
                    print("checking for similarities")
                    if self._is_similar(img, old_img):
                        print("Same Image")

                old_img = img.copy()

            for id in list_of_categories:
                new_json["categories"].append(self.new_category(int(id), str(id)))

            ############################## END OF XML FILE PARSING ####################################

        else:
            ###################################################### NON-XML FILE PARSING ##########################

            # Need to check the bounding boxes of this one!

            with open(path_to_mardct, 'r') as file:
                file_contents = file.read()

            list_of_categories = []
            image_id_counter = 0
            ann_id_counter = 0
            file_contents = file_contents.split("#")
            for block in file_contents:
                block = block.split("\n")
                block = list(filter(None, block))
                
                if len(block) <= 0:
                    continue

                frame_number = int(block[0])
                img = frame_to_image[frame_number]

                new_json["images"].append(self.entry_img(video_name+ str(frame_number), img.shape[0], img.shape[1], image_id_counter))
                image_id_counter += 1


                start_of_ids_index = 1

                while True:
                    try:
                        class_id = int(block[start_of_ids_index])

                        if class_id not in list_of_categories:
                            list_of_categories.append(class_id)

                        x0 = int(block[start_of_ids_index + 1].split(" ")[0])
                        y0 = int(block[start_of_ids_index + 1].split(" ")[1])
                        x1 = int(block[start_of_ids_index + 2].split(" ")[0])
                        y1 = int(block[start_of_ids_index + 2].split(" ")[1])
                        start_of_ids_index += 3

                        new_json["annotations"].append(self.entry_bbox([x0, y0, (x1-x0), (y1-y0)], class_id, image_id_counter, ann_id_counter))
                        ann_id_counter += 1
                    except:
                        start_of_ids_index = 1
                        break

            for id in list_of_categories:
                new_json["categories"].append(self.new_category(int(id), str(id)))

            ############################################### END OF NON-XML FILE PARSING ##########################


        try:
            self.clear()
        except:
            print("No internal data found.  Setting data from new file.")

        self._json_data = new_json

        print("Mardct data successfully loaded in to this current instance. \n Please remember when saving"\
                " to include a save path.  Using the method save_internal_json(save_name= your/path.json) is recommended")

        return self

    def extend_data(self, json_path=None, json_data=None, image_folder_path=None, save=True):
        """
        This method extends the data inside the current CVJ object with the supplied data.
        
        Parameters
        ----------
        json_path : string
            (Default = None)
             This is the path to the json file you want to extend the internal cvj data with.
            
        json_data : CVJ or Dict
            (Default = None)
            This is another way to extend your dataset.  This variable uses the data you supply that has
            already been loaded rather than loading from a file
        
        Returns
        -------
        CVJ : CVJ
            This returns the CVJ object.

        """
        self.clear_dicts()
        id_counter = 0

        if json_path != None:
            cvj_obj = CVJ(json_path=json_path)

            ## some images might not have annotations and therefore problems occur using the ids gathered from annotations
            ## which is what image_id_2_anns does.

            ## Need to check if the same name of image is in either json file
            img_ids = self.get_image_ids()
            img_ids_2 = cvj_obj.get_image_ids()
            img_id_2_anns = cvj_obj.get_image_id_2_anns()

            number_of_images = len(self._json_data["images"]) + len(cvj_obj["images"])

            max_internal_id = max(img_ids)

            img_id_map = {}
            for id in img_ids_2:
                id_counter += 1
                img_id_map[id] = max_internal_id + id_counter
            
            
            for id, value in img_id_2_anns.items():
                for ann in value:
                    ann["image_id"] = img_id_map[ann["image_id"]]

            for img in cvj_obj["images"]:
                img["id"] = img_id_map[img["id"]]


            ####This is so files with the same name can still be inserted in to the json
            list_of_datas = [self, cvj_obj]

            img_name_to_new_name = {}
            img_count = 0
            for dataset in list_of_datas:
                for img in dataset["images"]:
                    extension = os.path.splitext(img["file_name"])[1]
                    img_name_to_new_name[img["file_name"]] = str(img_count) + extension
                    img["file_name"] = str(img_count) + extension
                    img_count += 1
            ####This is so files with the same name can still be inserted in to the json


            self._json_data["images"].extend(cvj_obj["images"])
            self._json_data["categories"].extend(cvj_obj["categories"])
            self._json_data["annotations"].extend(cvj_obj["annotations"])
        elif json_data != None:
            self._json_data["images"].extend(json_data["images"])
            self._json_data["categories"].extend(json_data["categories"])
            self._json_data["annotations"].extend(json_data["annotations"])



        if image_folder_path != None and self._image_folder_path != None:
            path_name = os.path.basename(self._image_folder_path) + "_with_" + os.path.basename(image_folder_path)
            subprocess.call(['mkdir', '-p', path_name])

            images_1 = os.listdir(image_folder_path)
            images_2 = os.listdir(self._image_folder_path)

            images_1 = [os.path.join(image_folder_path, img) for img in images_1]
            images_2 = [os.path.join(self._image_folder_path, img)for img in images_2]

            images_1.extend(images_2)

            for img in tqdm(images_1, desc="Creating an image directory at {}".format(path_name)):
                basename = os.path.basename(img)
                shutil.copy2(img,os.path.join(path_name, basename))
                os.rename(os.path.join(path_name, basename), os.path.join(path_name, img_name_to_new_name[basename]))

        if save:
            self.save_internal_json(save_name=path_name + ".json")

            self.json_path = path_name + ".json"
            self.image_folder_path = path_name

            
        return self



                    



                


