from ..cvj import CVJ  
import cv2
import copy
import os
import json
from tqdm import tqdm

class Image_Tester(CVJ):
    """
    The Image_Tester class takes the json Handle
    and can perform checks and clean the images

    Parameters
    ----------
 
    Returns
    -------

    """
    def __init__(self, cvj, image_folder_path=None):
        super().__init__()

        if image_folder_path != None:
            self.image_folder_path = image_folder_path # Calling the setter

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

    def remove_unreadable_files(self, save=False):
        """
        This method just scans the images supplied to the object
        with the image folder path using opencv.  It goes through and
        opens each file and if there is an error opening the file then it
        removes that from the json file and moves the unreadable file
        
        NOTE: The path needs to be associated with the json path that was supplied for this to work.

        Parameters
        ----------
        None :
            

        Returns
        -------

        """

        image_id_2_filepath = self.get_image_id_2_filepath()
        image_id_2_anns = copy.deepcopy(self.get_image_id_2_anns())
        
        image_id_to_image_attribs = self.get_image_id_2_image_attribs()

        categories = copy.deepcopy(self._json_data["categories"])
        new_json = self.create_empty_json()

        keys = list(image_id_2_filepath.keys())
        deleted_keys = []
        deleted_filenames = []
        for key in tqdm(keys):
            filepath = image_id_2_filepath[key]
            
            try:
                
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                
                rows = img.shape[0]
                cols = img.shape[1]
                
            except:
                print("Exception occurred")
                if not os.path.isdir("opencv_unreadable_images"):
                    os.makedirs("opencv_unreadable_images")

                name = os.path.basename(filepath)
                deleted_filenames.append(name)
                new_file_path = os.path.join("opencv_unreadable_images", name)
                os.rename(filepath, new_file_path)
                print("deleting image {} from json at {}".format(os.path.basename(image_id_2_filepath[key]), image_id_2_filepath[key]))
                deleted_keys.append(key)
                del image_id_2_anns[key]
            
            
        anns = []
        for img_id in image_id_2_anns:
            anns.extend(image_id_2_anns[img_id])

        images = []
        for img_id in image_id_2_anns:
           images.append(image_id_to_image_attribs[img_id])

        new_json["categories"] = categories
        new_json["annotations"] = anns
        new_json["images"] = images

        self._json_data = new_json
        self._internal_clearing()

        if save:
            print("Saving new json")
            with open(self._json_path, 'w') as file:
                json.dump(new_json, file)
            if os.path.isdir("opencv_unreadable_images"):
                with open("opencv_unreadable_images/deleted_keys.txt", 'w') as file:
                    file.write(str(deleted_keys))

        return deleted_keys, deleted_filenames
