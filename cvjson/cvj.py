"""
This script creates a handle object for json
in the COCO format.  The aim for this handle is
to make redundant code less redundant, safer, 
easy to use, and data extraction very simple.

This api uses CVJ as the super class
and all other subclasses are meant to 
extend the functionality.

Structure this library uses is as follows.

.. code-block:: python

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

Author: Benjamin Anderson Garrard
"""

import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

 
class CVJ():
    """
    The CVJ class is the most basic class and will
    only give information based on the current json file supplied.
    This means that regarding purely the json file and accompanying files, images,
    etc.  This will describe that data or help generate the information in to usable
    data.  Anything else that is outside gaining insight or gathering data from the json will
    be in the form of an extension.

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

    # get_dictionary(cvj_enum=cvj.IMID_2_ANNS) range= 0-7
    IMID_2_ANNS = 0
    CLID_2_NAME = 1
    CLNAME_2_CLID = 2
    IMID_2_FNAME = 3
    FNAME_2_IMID = 4
    IMID_2_FPATH = 5
    IMID_2_IMATTR = 6
    CLID_2_ANNS = 7

    # creating negative category id based on phone keypad and the letters in
    #  negative 63428483
    NEGATIVE_CLASS = 63428483

    def __init__(self, json_path=None, image_folder_path=None):
        self._json_path = None
        self._json_data = None 
        self._image_folder_path = None
        self._image_class_counts_path = None
        self._class_to_filepath_data = None
        self._image_id_2_anns_dict = None
        self._class_id_2_name_dict = None
        self._class_name_2_id_dict = None
        self._img_id_2_filename_dict = None
        self._filename_2_image_id_dict = None
        self._imageid_to_filepath_dict = None
        self._image_id_2_image_attribs = None
        self._class_id_2_anns_dict = None

        if json_path is not None:
            self.json_path = json_path
            self._json_data = self.load_json(json_path)

        if image_folder_path is not None:
            self.image_folder_path = image_folder_path

    def __setitem__(self, key, item):
        self._json_data[key] = item

    def __getitem__(self, key):
        return self._json_data[key]

    def __repr__(self):
        return repr(self._json_data)

    def __len__(self):
        return len(self._json_data)

    def __delitem__(self, key):
        del self._json_data[key]

    def __iter__(self):
        return (key for key in self._json_data.keys())

    def __contains__(self, key):
        return key in self._json_data

    def clear(self):
        """
        This function will clear everything from the object.
        This includes all built dictionaries previously navigated with
        other data.
        """
        self._json_path = None
        self._image_folder_path = None
        self._image_class_counts_path = None
        self._class_to_filepath_data = None

        self._image_id_2_anns_dict = None
        self._class_id_2_name_dict = None
        self._class_name_2_id_dict = None
        self._img_id_2_filename_dict = None
        self._filename_2_image_id_dict = None
        self._imageid_to_filepath_dict = None
        self._image_id_2_image_attribs = None
        self._class_id_2_anns_dict = None
        return self._json_data.clear()

    def copy(self):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.copy()

    def has_key(self, k):
        """Function that is exactly the same as found in a dict."""
        return k in self._json_data

    def update(self, *args, **kwargs):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.update(*args, **kwargs)

    def keys(self):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.keys()

    def values(self):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.values()

    def items(self):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.items()

    def pop(self, key):
        """Function that is exactly the same as found in a dict."""
        return self._json_data.pop(key)

    def setdefault(self, key, value=None):
        """Function that is exactly the same as found in a dict."""
        if value is None:
            return self._json_data.setdefault(key)
        else:
            return self._json_data.setdefault(key, value)

    def clear_dicts(self):
        """Clear all saved states of dicts in memory."""
        self._image_id_2_anns_dict = None
        self._class_id_2_name_dict = None
        self._class_name_2_id_dict = None
        self._img_id_2_filename_dict = None
        self._filename_2_image_id_dict = None
        self._imageid_to_filepath_dict = None
        self._image_id_2_image_attribs = None
        self._class_id_2_anns_dict = None

    @property
    def json_path(self):
        return self._json_path

    @json_path.setter
    def json_path(self, path):
        """
        Set the path to the JSON dataset file.

        Parameters
        ----------
        path : string
             The path must be a COCO formatted JSON similar to what is found
             at the top of this script.

        Example Usage
        -------------
        .. code:: python

            json_handle.json_path = "path/to/file.json"
        """

        if os.path.isfile(path):
            self._json_path = path
            self._json_data = self.load_json(self.json_path)
            self._image_id_2_anns_dict = None
            self._class_id_2_name_dict = None
            self._class_name_2_id_dict = None
            self._img_id_2_filename_dict = None
            self._filename_2_image_id_dict = None
            self._imageid_to_filepath_dict = None
            self._image_id_2_image_attribs = None
            self._class_id_2_anns_dict = None
            try:
                self._json_data["images"]
                self._json_data["annotations"]
                self._json_data["categories"]
            except KeyError as e:
                print("\n\nYou have input an ill formatted COCO formatted JSON. "
                      "The error was a nonexistent key = {}".format(e))
                raise
        else:
            print("\n\nThe path supplied is not a file\n\n")
            print("path supplied = {}\n\n".format(path))

    @property
    def image_folder_path(self):
        return self._image_folder_path

    @image_folder_path.setter
    def image_folder_path(self, path):
        """
        Set the path to the image folder that corresponds to the given JSON
        dataset file.

        Parameters
        ----------
        path : string
             The path to the images of the dataset.

        Example Usage
        -------------
        .. code:: python

            json_handle.image_folder_path = "path/to/images"
        """

        if path is not None:
            if os.path.isdir(path):
                self._image_folder_path = path
                self._imageid_to_filepath_dict = None
            else:
                print("The path supplied is not a file directory")
                print(path)

    @property
    def image_class_counts_path(self):
        return self._class_to_filepath_path

    @image_class_counts_path.setter
    def image_class_counts_path(self, path):
        """
        This is a very interesting attribute.  This attribute
        can only be generated when you use the Cropper Class
        in conjunction with the json Handle.  Which the Cropper
        actually generates a file called image_class_counts.json.
        That file actually contains the filepaths to the images
        rather than the count.

        TODO: Change the "image_class_counts.json" to something more intuitive

        Parameters
        ----------
        path : string
              The file path that this is referring to is the file that is 
              called image_class_counts.json.

        Example Usage
        -------------
        .. code:: python

            json_handle.class_to_filepath_path = "path/to/filepath.json"
        """

        if os.path.isfile(path):
            self._class_to_filepath_path = path
            self._class_to_filepath_data = self.load_json(path)

        else:
            print("The path supplied is not a file")
            print(path)

    def load_json(self, path):
        """
        Convert path to JSON file to a dictionary and return it.  This method
        does not store the data.  Use the setter method of CVJ to do so.

        Parameters
        ----------
        path: string
            This path must be to any valid json file.

        Returns
        -------
        dict: dict
            Loaded JSON data.
        """
        assert os.path.isfile(path), path
        with open(path, 'r') as infile:
            jsdat = json.load(infile)
        return jsdat

    def get_image_ids(self):
        """
        Return a list of all image IDs in the CVJ object.

        Returns
        -------
        list: list
            A list of image ids.
        """
        return list(self.get_image_id_2_image_attribs().keys())

    def get_filenames(self):
        """
        Return a list of all filenames in the CVJ object.

        Returns
        -------
        list: list
            A list of filenames.
        """
        return list(self.get_filename_2_image_id().keys())

    def get_annotations(self):
        """
        Return all annotations found in the CVJ object.

        Returns
        -------
        list: list
            A list of annotations.
        """
        return self._json_data["annotations"]

    def get_category_ids(self):
        """
        Return a list of all the category IDs in the CVJ object.

        Returns
        -------
        list: list
            A list of category IDs.
        """
        return list(self.get_class_id_2_name().keys())

    def get_category_names(self):
        """
        Return a list of all category names in the CVJ object.

        Returns
        -------
        list: list
            A list of all category names.
        """
        return list(self.get_class_id_2_name().values())

    def get_dictionary(self, cvj_enum: int) -> dict:
        """
        Return internal dictionary based on the enum given.

        Parameters
        ----------
        cvj_enum: int
            An integer corresponding to a dictionary.  Named values are
            available as psuedo-enums
            CVJ.IMID_2_ANNS = Image ID to Annoations
            CVJ.CLID_2_NAME = Class ID to Class Name
            CVJ.CLNAME_2_CLID = Class name to Class ID
            CVJ.IMID_2_FNAME = Image ID to File Name
            CVJ.FNAME_2_IMID = File Name to Image ID
            CVJ.IMID_2_FPATH = Image ID to File Path
            CVJ.IMID_2_IMATTR = Image Id to Image Attributes
            CVJ.CLID_2_ANNS = Class ID to Annotations

        Returns
        -------
        dict
            A dictionary that corresponds to the passed parameter.
        """

        if cvj_enum == CVJ.IMID_2_ANNS:
            return self.get_image_id_2_anns()
        elif cvj_enum == CVJ.CLID_2_NAME:
            return self.get_class_id_2_name()
        elif cvj_enum == CVJ.CLNAME_2_CLID:
            return self.get_class_name_2_id()
        elif cvj_enum == CVJ.IMID_2_FNAME:
            return self.get_image_id_2_filename()
        elif cvj_enum == CVJ.FNAME_2_IMID:
            return self.get_filename_2_image_id()
        elif cvj_enum == CVJ.IMID_2_FPATH:
            return self.get_image_id_2_filepath()
        elif cvj_enum == CVJ.IMID_2_IMATTR:
            return self.get_image_id_2_image_attribs()
        elif cvj_enum == CVJ.CLID_2_ANNS:
            return self.get_class_id_2_anns()

    def get_image_id_2_anns(self, img_id=None) -> (dict or None, dict):
        """
        Return annotations associated with the given image id.  Also return
        a dictionary containing all of the image ids and their annotations.

        Parameters
        ----------
        img_id : int
            (Default = None)
            This is the image ID for an image that is in the JSON data of the
            current CVJ object.

        Returns
        -------
        dict:
            A dict with the image ID supplied and it's corresponding
            annotations.
        dict:
            The internal dict has has all of the image IDs and their
            corresponding annotations associated with those IDs.
        """
        if self._image_id_2_anns_dict is None:
            imgid2anns = defaultdict(list)
            for ann in self._json_data["annotations"]:
                imgid2anns[ann["image_id"]].append(ann)

            self._image_id_2_anns_dict = dict(imgid2anns)

        if img_id is not None:
            return ({img_id: self._image_id_2_anns_dict[img_id]},
                    self._image_id_2_anns_dict)

        return None, self._image_id_2_anns_dict

    def get_class_id_2_name(self, class_id=None):
        """
        Return class names associated with the given class id.  Also return
        a dictionary containing all of the class ids and their annotations.

        Parameters
        ----------
        class_id : int
            (Default = None)
            This is the class ID for a class that is in the JSON data of the
            the current CVJ object.

        Returns
        -------
        dict:
            A dict with the class ID supplied and it's corresponding
            class name.
        dict:
            The internal dict has has all of the class IDs and their
            corresponding class names associated with those class IDs.
        """
        if self._class_id_2_name_dict is None:
            id2name = defaultdict(list)
            for cat in self._json_data["categories"]:
                id2name[cat["id"]].append(cat)

            self._class_id_2_name_dict = dict(id2name)

        if class_id is not None:
            return ({class_id: self._class_id_2_name_dict[class_id]},
                    self._class_id_2_name_dict)

        return None, self._class_id_2_name_dict

    def get_class_name_2_id(self, class_name=None, json_data=None):
        """
        This method creates a dictionary using the category name as the key and the
        category id's are the values.
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------
        class_name : string
            (Default = None)
            This is the class name for a class that is in the JSON data of the object
            or the supplied JSON data from the json_data variable.

        json_data : dict
             (Default value = None)
             This is the loaded data from a COCO formatted JSON file.
                * If this is supplied all data returned will be from this variable.

        Returns
        -------
        int : int
            If class_name is supplied then this method returns the class ID.

        dict : dict
            This is only returned if there is no class_name supplied to the method.
                * keys   = class names like "bear", "car", "alien", "person", etc

                * values = the category ids or also known as the class ids.  The number that represents the class.
        """

        name2id = {}
        if json_data == None:
            if self._class_name_2_id_dict != None:
                if class_name == None:
                    return self._class_name_2_id_dict
                else:
                    return self._class_id_2_name_dict[class_name]

            else:
                for category in self._json_data["categories"]:
                    key = category["name"].lower()
                    if key not in name2id:
                        name2id[key] = category["id"]

                self._class_name_2_id_dict = name2id
        
        else:
            for category in json_data["categories"]:
                key = category["name"].lower()
                if key not in name2id:
                    name2id[key] = category["id"]

        if class_name == None:
            return name2id
        else:
            return name2id[class_name]

    def get_image_id_2_filename(self, img_id=None, json_data=None):
        """
        This method creates a dictionary using the image id as the key and the values
        are the filenames associated with the image id.
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------

        img_id : int
            (Default = None)
            This is the image ID for an image that is in the JSON data of the object
            or the supplied JSON data from the json_data variable.

        json_data : dict
            (Default value = None)
            This is the loaded data from a COCO formatted JSON file.
            * If this is supplied all data returned will be from this variable.

        Returns
        -------
        string : string
            If the img_id is supplied then this method will return the file name associated 
            with that image id.  

        dict : dict
             This is only returned if there is no img_id supplied to the method.
                * keys   = image ids

                * values = filenames with the extension so the will have ".png", ".tif", or something similar

        """
        image_id2_filename = {}
        if json_data == None:
            if self._img_id_2_filename_dict != None:
                if img_id == None:
                    return self._img_id_2_filename_dict
                else: 
                    return self._img_id_2_filename_dict[img_id]
            else:
                
                for image in self._json_data["images"]:
                    image_id2_filename[image["id"]] = image["file_name"]

                self._img_id_2_filename_dict = image_id2_filename
        else:
            for image in json_data["images"]:
                image_id2_filename[image["id"]] = image["file_name"]

            if img_id == None:
                return image_id2_filename

            else: 
                return image_id2_filename[img_id]

        if img_id == None:
            return image_id2_filename

        else: 
            return image_id2_filename[img_id]

        return image_id2_filename

    def get_filename_2_image_id(self, filename=None, json_data=None):
        """
        This method creates a dictionary using the filename as the key and the image
        id as the value.
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------
        img_id : int
            (Default = None)
            This is the file name for an image that is in the JSON data of the object
            or the supplied JSON data from the json_data variable.

        json_data : dict
             (Default value = None)
             This is the loaded data from a COCO formatted JSON file.
                * If this is supplied all data returned will be from this variable.

        Returns
        -------
        int : int
            This is the image ID of the file name that was supplied to this method.  

        dict : dict
             This is only returned if there is no file name supplied to the method.
                * keys   = filenames with the extension so the will have ".png", ".tif", or something similar

                * values = image ids

        """

        filenamedict ={}
        if json_data == None:
            if self._filename_2_image_id_dict != None:
                if filename == None:
                    return self._filename_2_image_id_dict
                else:
                    return self._filename_2_image_id_dict[filename]

            else:  
                for image in self._json_data["images"]:
                    #print(image["file_name"])
                    filenamedict[image["file_name"]] = image["id"]

                    self._filename_2_image_id_dict = filenamedict
            
        else:

            for image in json_data["images"]:
                #print(image["file_name"])
                filenamedict[image["file_name"]] = image["id"]

        if filename == None:
            return filenamedict
        else:
            return filenamedict[filename]

    def get_image_id_2_filepath(self, img_id=None):
        """
        This method will not work unless an image filepath has been supplied.
        So first set the filepath like so:
        
        .. code-block:: python
        
            cvj_object.image_folder_path = /your/path/to/images
        
        This method creates a dictionary using the image id as the key and
        the filepaths associated with the image id as the value.
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------
        img_id : int
            (Default = None)
            This is the image ID for an image that is in the JSON data of the object
            or the supplied JSON data from the json_data variable.

        Returns
        -------

        string : string
            This is the filepath of the supplied Image ID

        dict : dict 
             This returns only if the img_id is not supplied to this method
                * keys   = image ids

                * values = the filepaths associated with each image id

        """
        is_missing = False
        image_id2_file_path_dict = {}

        if self._image_folder_path == None:
            print("Please set the image_folder_path variable before trying"\
                    " to access a method that requires those images.\n\n")
            return None
        else:
            if self._imageid_to_filepath_dict != None:
                if img_id == None:
                    return self._imageid_to_filepath_dict
                else:
                    return self._imageid_to_filepath_dict[img_id]
            else:

                assert os.path.isdir(self._image_folder_path), self._image_folder_path

                filename_2_image_id_dict = self.get_filename_2_image_id()

                
                for root, directs, files in os.walk(self._image_folder_path): 
                    for image in files:
                        if image.endswith((".png", ".jpg", ".jpeg", ".tif")):
                            try:
                                id = filename_2_image_id_dict[image]
                                image_id2_file_path_dict[id] = os.path.join(root, image)
                            except KeyError:
                                is_missing = True
                                print("could not find {} in json file".format(image))
                        else:
                            print("could not find image or image does not end with .jpg, .jpeg, .png, .tif")

                if is_missing:
                    print("the files missing are either in another format, don't exist, or are part of a train/validation split")


            self._imageid_to_filepath_dict = image_id2_file_path_dict


        if img_id == None:
            return image_id2_file_path_dict
        else:
            return image_id2_file_path_dict[img_id]

    def get_image_id_2_image_attribs(self, img_id=None, json_data=None):
        """
        This method creates a dictionary using the image id as the key and
        the attributes of that image as the value
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------

        img_id : int
            (Default = None)
            This is the image ID for an image that is in the JSON data of the object
            or the supplied JSON data from the json_data variable.

        json_data : dict
             (Default value = None)
             This is the loaded data from a COCO formatted JSON file.
                * If this is supplied all data returned will be from this variable.

        Returns
        -------

        dict : dict
            If the image id is supplied to the img_id variable then this method returns
            a dict with the attributes of the image.  For more information on the format
            of the dictionary returned look at the top of this script or refer to the official
            documentation page here -> https://bengarrard.bitbucket.io/ and look for 
            "Introduction to the CVJ".

        dict : dict
            This returns only if the img_id is not supplied to this method
                * keys   = image ids

                * values = image attributes associated with each image id

        """
        img_attribs_dict = {}
        if json_data == None:
            if self._image_id_2_image_attribs != None:
                if img_id == None:
                    return self._image_id_2_image_attribs
                else:
                    return self._image_id_2_image_attribs[img_id]

            else:
                images = self._json_data["images"]

                for img in images:
                    img_attribs_dict[img["id"]] = img

                self._image_id_2_image_attribs = img_attribs_dict
        else:
            images = json_data["images"]
            for img in images:
                img_attribs_dict[img["id"]] = img

        if img_id == None:
            return img_attribs_dict
        else:
            return img_attribs_dict[img_id]




    def get_distribution_of_class_id(self, show_plot=True):
        """
        This method makes a list of the category id's from the
        annotations and just appends them.  
        
        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------

        Example
        -------

            So an example of
            what the list could look like is:
        
            [1,1,2,3,4,5,5,5,5]
            
            Now if the show_plot parameter is equal to True.
            Then seaborn will plot it and wait for input.
        """

        image_id_2_anns = self.get_image_id_2_anns()

        classes = []
        for id in image_id_2_anns:
            for ann in image_id_2_anns[id]:
                classes.append(ann["category_id"])
        
        if show_plot:
            sns.distplot(classes, rug=False, kde=True)
            #plt.savefig("class_dist.png")
            plt.show()
            plt.close()

        return classes

    def get_distribution_of_area(self, show_plot=True):

        """
        This method makes a list of the areas from the
        annotations and just appends them.  
    
        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------
        list: list
             A list of appended areas of the bounding boxes.

        Example
        -------

            So an example of
            what the returned list could look like is:

            [100,100,2000,3000,4000,2405,500,50,500]

                
        Now if the show_plot parameter is equal to True.
        Then seaborn will plot it and wait for input.

        """
        image_id_2_anns = self.get_image_id_2_anns()

        area = []
        for id in image_id_2_anns:
            for ann in image_id_2_anns[id]:
                area.append(ann["area"])
        
        sns.distplot(area, rug=False, kde=True)
        plt.show()
        plt.close()

        return area


    def get_average_side_lengths(self, show_plot=True):

        """
        This method makes a list of the areas from the
        annotations and just appends them.  
    
        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------
        list: list
            A list of the square root of the areas.  
                
        Now if the show_plot parameter is equal to True.
        Then seaborn will plot it and wait for input.

        """
        image_id_2_anns = self.get_image_id_2_anns()

        sides = []
        for id in image_id_2_anns:
            for ann in image_id_2_anns[id]:
                sides.append(np.sqrt(ann["area"]))
        
        sns.distplot(sides, rug=False, kde=True)
        plt.show()
        plt.close()

        return sides

    def get_average_area_by_class(self, show_plot=True):
        """
        
        This method will get the average area for each class
        and returns the values in a dictionary.
        If the show_plot param is true then it plots the
        dictionary.
        

        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------
        dict : dict
            * Keys   = class ids

            * values = average areas associated with each class id

        """
        image_id_2_anns = self.get_class_id_2_anns()
        
        x = list(image_id_2_anns.keys())
        y = []
        print("Getting Averages")
        if show_plot:
            for id in image_id_2_anns:
                areas = []
                for ann in tqdm(image_id_2_anns[id]):
                    areas.append(ann["area"])

                y.append(np.asarray(areas).mean())
            sns.barplot(x=x, y=y)
            plt.show()
            plt.close()
        else:
            for id in image_id_2_anns:
                areas = []
                for ann in image_id_2_anns[id]:
                    areas.append(ann["area"])

                y.append(np.asarray(areas).mean())
        
        class_averages = {}
        i = 0
        for id in x:
            class_averages[id] = y[i]
            i+=1

        return class_averages

    def get_class_id_2_anns(self, class_id=None, json_data=None, verbose=False):
        """
        This method returns a dictionary that has the class id
        as the key and the annotations to that class as the values.
        
        If there is already one created it just returns the previously made one
        to improve performance.

        Parameters
        ----------
        class_id : int
            (Default = None)
            This is the class ID for the annotations associated with that class.

        json_data : dict
             (Default value = None)
             This is the loaded data from a COCO formatted JSON file.
                * If this is supplied all data returned will be from this variable.

        verbose :bool
             (Default value = False)

        Returns
        -------

        dict : dict
        This returns only if the img_id is not supplied to this method
            * keys   = class ids

            * values = annotations associated with each class id

        """
        class_id_2_anns_dict = {}
        if json_data == None:
            if self._class_id_2_anns_dict != None:
                if class_id == None:
                    return self._class_id_2_anns_dict
                else:
                    return self._class_id_2_anns_dict[class_id]

            else:

                print("Getting annotations by id")
                i = 0
                for ann in self._json_data["annotations"]:
                    if ann["category_id"] not in class_id_2_anns_dict:
                        class_id_2_anns_dict[ann["category_id"]] = [ann,]
                    else:
                        class_id_2_anns_dict[ann["category_id"]] += [ann,]

                    if verbose:
                        if i % 1000 == 0:
                            print("Gathering annotations by id iteration {}.".format(i))

                        i += 1

                self._class_id_2_anns_dict = class_id_2_anns_dict

        else:
            
            print("Getting annotations by id")
            i = 0
            for ann in json_data["annotations"]:
                if ann["category_id"] not in class_id_2_anns_dict:
                    class_id_2_anns_dict[ann["category_id"]] = [ann,]
                else:
                    class_id_2_anns_dict[ann["category_id"]] += [ann,]

                if verbose:
                    if i % 10000 == 0:
                        print("Gathering annotations by id iteration {}.".format(i))

                    i += 1

        if class_id == None:
            return class_id_2_anns_dict
        else:
            return class_id_2_anns_dict[class_id]

    def create_json_by_class(self, list_of_ids, verbose=True):
        """
        This method creates a json based on the class
        id's supplied.  The json will only have annotations
        for those classes.
        
        The json is not saved in this method

        Parameters
        ----------
        list_of_ids : list
             A list of class ids to be included in to the json dictionary

        verbose : bool
             (Default value = False)
             This prints out a verbose message of what iteration count it is at when it
             is searching through the json through the annotations.
            

        Returns
        -------
        dict : dict
            * keys   = class ids

            * values = annotations associated with each class
        """

        new_json = self.create_empty_json()
        id_2_anns = self.get_class_id_2_anns(verbose=verbose)

        for id in list_of_ids:
            new_json["annotations"].extend(id_2_anns[id])

        new_json["images"] = self._json_data["images"]
        new_json["categories"] = [self.categ_idx_to_coco_categ(id) for id in list_of_ids]

        return new_json


    def create_json(self,new_json, save_path=None):
        """
        This method creates a json file from a dictionary that is
        supplied. If no save path is supplied then it creates a file
        in the folder containing the json path supplied to the object
        and the file name will be "_new_json_DEFAULT.json"

        Parameters
        ----------
        new_json : dict
        
        save_path : string
            (Default value = None)
            Needs to be a path with a file name

        Returns
        -------
        dict : dict
            * The same dict that was supplied

        """
        if save_path != None:
            dir_path = os.path.dirname(save_path)

            if os.path.isdir(dir_path):
                with open(save_path, 'w') as file:
                    json.dump(new_json, file)
                print("file saved at {}".format(save_path))
            else:
                print("Save path does not come from a real directory. Path supplied = {}, nonexistent directory = {}".format(save_path, dir_path))
        else:
            dir_name = os.path.dirname(self._json_path)
            file_name = dir_name + "/_new_json_DEFAULT.json"
            with open(file_name, 'w') as file:
                json.dump(new_json, file)
            print("No save path given defaulting to json path direcory {}".format(dir_name))

        return new_json

    def get_count_files_by_class(self, verbose=False, show_plot=False):

        """
        This method is only used with the image_class_count.json file generated by the Cropper
        class.  This just shows how many files were made for each class.  If cropping
        to bounding box center was used then it will either have the same amount of images
        for each class as there is bounding boxes for each class or more through augments.

        Parameters
        ----------
        verbose : bool
            (Default value = False)
            This parameter has the console output information during it gathering the data.  The verbose will look similar to
            "Class ID 5 has 280 images"

        show_plot : bool
            (Default value = False)
            This parameter when set to true generates a bar plot showing the class id on the x axis

        Returns
        -------
        dict : dict
            * keys   = class ids

            * values = image counts.  (How many images are associated with the class id)
        """
        if self._image_class_counts_path != None and self._class_to_filepath_data == None:
            self.image_class_counts_path = self._image_class_counts_path

        class_id_to_filepaths = self._class_to_filepath_data

        count_dict = {}

        if verbose:
            for id in class_id_to_filepaths:
                print("Class ID {} has {} images.".format(id, len(class_id_to_filepaths[id])))
                count_dict[id] = len(class_id_to_filepaths[id])
        else:
            for id in class_id_to_filepaths:
                count_dict[id] = len(class_id_to_filepaths[id])

        if show_plot:
            print("Plotting")
            x = list(count_dict.keys())
            y = list(count_dict.values())
        
            ax = sns.barplot(x=x,y=y)
            ax.set_title("Image Counts vs Class ID")
            plt.show()
            plt.close()

        return count_dict

    
    def get_class_count_by_img_id(self, img_id, show_plot=True):
        """
        This method will count how many examples of bounding boxes exist for each
        class on the supplied image id. If the show_plot variable is true then it
        generates a bar plot for the image id and the puts the classes on the x axis
        and the counts on the y axis
        
        NOTE:
        The image id supplied must be a part of the json file that is stored within
        the object.

        Parameters
        ----------
        img_id : int
            This parameter is the image id of the img the user wants to find out 
            how many bounding boxes are on the image and what there classes are 
            
        show_plot : bool
            (Default value = True)
            This parameter when set to true generates a bar plot showing the class ids on the x axis

        Returns
        -------
        dict : dict
            * keys   = class ids

            * values = count of annotations for each of the class ids

        """
        image_id_2_anns = self.get_image_id_2_anns()

        class_counts = {}
        for ann in image_id_2_anns[img_id]:
            if ann["category_id"] not in class_counts:
                class_counts[ann["category_id"]] = [ann,]
            else:
                class_counts[ann["category_id"]] += [ann,]

        for id in class_counts:
            class_counts[id] = len(class_counts[id])

        if show_plot:
            print("Plotting")
            x = list(class_counts.keys())
            y = list(class_counts.values())
        
            ax = sns.barplot(x=x,y=y)
            ax.set_title("Image ID {}".format(str(img_id)))
            plt.show()
            plt.close()

        return class_counts

    def get_class_count_by_filename(self, filename, show_plot=True):
        """
        This method will count how many examples of bounding boxes exist for each
        class is on the supplied filename.
        
        NOTE:
        The filename supplied must be a part of the json file that is stored within
        the object.

        Parameters
        ----------
        filename : string
            This parameter is the filename of the image the user wants to find
            out how many bounding boxes are on the image and what there classes are 

        show_plot : bool
            (Default value = True)
            This parameter when set to true generates a bar plot showing the class id on the y axis

        Returns
        -------
        dict : dict
            * keys   = class ids

            * values = count of annotations for each of the class ids

        """
        image_id_2_anns = self.get_image_id_2_anns()
        filename_2_image_id_dict = self.get_filename_2_image_id()

        img_id = filename_2_image_id_dict[filename]

        class_counts = {}
        for ann in image_id_2_anns[img_id]:
            if ann["category_id"] not in class_counts:
                class_counts[ann["category_id"]] = [ann,]
            else:
                class_counts[ann["category_id"]] += [ann,]

        for id in class_counts:
            class_counts[id] = len(class_counts[id])

        if show_plot:
            print("Plotting")
            x = list(class_counts.keys())
            y = list(class_counts.values())
        
            sns.barplot(x=x,y=y)
            plt.show()
            plt.close()

        return class_counts
    
    def get_max_counts_per_img(self, show_plot=True):
        """
        This method will plot the most demanding image for cropping each bounding box.
        This method goes through each image and counts the bounding boxes corresponding to the image.
        It then stores the maximum count of annotations for a class for that image in a dictionary with
        the key as the img id.
        
        This ends up being that each img_id will show the maximum count of a class out of all
        classes within each image.  This will be plotted using seaborn.

        To be quite honest I don't think that the chart is very useful, however the returned data can be.
    
        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------
        dict: dict
            *keys   = image ids

            *values = the counts of the most prominent class on each image 

        list : list
            The list of the classes that are in the same order as the keys in the returned dictionary


        Example
        -------
        
        If I have img_id 1 and I want to know which class is the most dominant in this image
        then I just simply call this method like below

        .. code-block:: python
            
            from cvj import CVJ

            cvj_object = CVJ(json_path)
            image_id_2_class_counts, classes = cvj_object.get_max_counts_per_img(show_plot=False)

            i = 0
            for image_id, class_count in image_id_2_class_counts.items():
                print("The img_id {} has class {} as the most dominant class with {} annotations".format(class_count, classes[i], class_count))
                i += 1


        Then in the plot I just look at the x axis and find the number 1 and then see what class is 

        """
        image_id_2_anns = self.get_image_id_2_anns()
        
        img_id_counts = {}
        classes = []
        for img_id in image_id_2_anns:
            class_counts = self.get_class_count_by_img_id(img_id, show_plot=False)
            max_id = max(class_counts, key=class_counts.get)
            img_id_counts[img_id] = class_counts[max_id]
            classes.append(max_id)
            print("Max Bounding Boxes = {}, img_id = {}, class_id = {}".format(class_counts[max_id],img_id, max_id))

        if show_plot:
            print("Plotting")
            x = list(img_id_counts.keys())
            y = list(img_id_counts.values())
        
            ax = sns.barplot(x=x,y=y)
            ax.set_title("Class Count vs Image ID")
            ax.legend(classes, loc="upper right")
            plt.show()
            plt.close()

        return img_id_counts, classes

        
    def get_class_id_2_anns_count(self, show_plot=True):
    
        """
        This function gets the count of bboxes by class ID. If show_plot is True (Default)
        then this will have a seaborn barchart pop up.

        Parameters
        ----------
        show_plot : bool
             (Default value = True)

        Returns
        -------
        dict: dict
            * keys   = class ids

            * values = count of annotations for each of the class ids
        """

        category_id_2_anns_dict = self.get_class_id_2_anns()

        bbox_count_by_class_dict = {}
        for id in category_id_2_anns_dict:
            bbox_count_by_class_dict[id] = len(category_id_2_anns_dict[id])

        if show_plot:
            print("Plotting")
            x = list(bbox_count_by_class_dict.keys())
            y = list(bbox_count_by_class_dict.values())
        
            ax = sns.barplot(x=x,y=y)
            ax.set_title("Bounding Boxes vs Class ID")
            plt.show()
            plt.close()


        return bbox_count_by_class_dict

    def _get_diameters(self, bboxes):
        '''
        This will take all of the bounding boxes and get the diameter of each on since
        bboxes[:,2] means every bounding box width and bboxes[:,3] means every bounding boxes hieght
        '''
        return np.sqrt(np.square(bboxes[:,2]) + np.square(bboxes[:, 3]))/np.sqrt(2)


    def xywh_to_xyxy(self, bboxes):
        """
        This method converts the bounding boxes of a numpy array in the format
        [[x, y, width, height]] to the format [[x1, y1, x2, y2]]


        Parameters
        ----------
        bboxes : numpy array
            This is the numpy array for bounding boxes in the format of [[x, y, width, height]]
            

        Returns
        -------
        numpy array : numpy array
            * This is the numpy array for bounding boxes in the format of [[x1, y1, x2, y2]]
        
        Example
        -------

        This code takes the bounding boxes in the
        form of a numpy array with the format 
        x, y, width, and height. For example

        .. code-block:: python

            x       = bboxes[:,0] #is all of the x's in the array
            y       = bboxes[:,1] #is all of the y's in the array
            widths  = bboxes[:,2] #is all of the w's in the array
            heights = bboxes[:,3] #is all of the h's in the array


        Directly below is somewhat how your array will have to look like

        .. code-block:: python

            [[ 24, 25, 4, 5],
            [50, 50, 7, 6]
            [....],
            [....]]

        """

        
        bboxes[:, 2] = bboxes[:,0] + bboxes[:,2]
        bboxes[:, 3] = bboxes[:,1] + bboxes[:,3]

        return bboxes

    def xyxy_to_xywh(self, xyxy):
        """
        This method converts the bounding boxes of a numpy array in the format
        [[x1, y1, x2, y2]] to the format [[x, y, width, height]]

        Parameters
        ----------
        xyxy : numpy array
            This is the numpy array for bounding boxes in the format of [[x1, y1, x2, y2]]
            

        Returns
        -------
        numpy array : numpy array
            * This is the numpy array for bounding boxes in the format of [[x, y, width, height]]

        """
        xyxy[:,2] -= xyxy[:,0] 
        xyxy[:,3] -= xyxy[:,1]
        return xyxy
        
    def create_json_of_class_focused_images(self, list_of_class_ids):
        """
        This method generates a dictionary in the COCO format json that this library
        uses from a file generated by the cropper known as "{TIMESTAMP}_image_class_counts.json".

        That file has the filepaths to each image in the "{TIMESTAMP}_coco_train.json" and the class
        the images that were created were based off of.

        Using that file, this method, given a list of ids, turns the selected class id's in to a new json
        that has the images most strongly associated with those IDs.  By strong I mean that class will be in the center
        of these images if the crop_images_bbox_centered() was used.  There could be more annotations for another class on the image.

        NOTE: This method does not save the dictionary.  The user must save it. create_json() can do the trick.

        Parameters
        ----------
        list_of_class_ids : list
             A list of class ids to single out for the new json

        Returns
        -------
        dict  : dict
             * keys   = "images", "annotations", "categories"
             * values = [image], [annotation], [category] 

        """
        new_json = self.create_empty_json()

        if self._image_class_counts_path != None and self._class_to_filepath_data == None:
            self.image_class_counts_path = self._image_class_counts_path

        if self._class_to_filepath_data != None:
            class_id_to_filepaths = self._class_to_filepath_data

            filename_to_image_id = self.get_filename_2_image_id()
            image_id_2_anns = self.get_image_id_2_anns()
            image_id_2_imgattribs = self.get_image_id_2_image_attribs()
            categories = []

            for id in list_of_class_ids:
                categories.append(self.categ_idx_to_coco_categ(id))

            anns = []
            images = []
            for id in list_of_class_ids:   
                try:
                    for filepath in class_id_to_filepaths[id]:
                        filename = os.path.basename(filepath)
                        image_id = filename_to_image_id[filename]
                        anns.extend(image_id_2_anns[image_id])
                        images.append(image_id_2_imgattribs[image_id])
                except KeyError:
                        for filepath in class_id_to_filepaths[str(id)]:
                            filename = os.path.basename(filepath)
                            image_id = filename_to_image_id[filename]
                            anns.extend(image_id_2_anns[image_id])
                            images.append(image_id_2_imgattribs[image_id])

            new_json["categories"] = categories
            new_json["annotations"] = anns
            new_json["images"] = images
            
        else:
            print("Please supply a json file with class ids and corresponding filepaths")

        return new_json

    def update_images(self, list_of_paths, remove=False):
        """
        This method updates the image annotations within the json_data that is stored within
        this CVJ object

        Parameters
        ----------
        list_of_paths : list
            This parameter is a list of paths to the images that the user is wanting to input
            in to the internal json data.

        Returns
        -------
        
        """

        ids = [int(id) for id in self.get_image_ids()]
        max_id = max(ids)

        if not remove:
            for path in list_of_paths:
                max_id += 1
                name = os.path.basename(path)
                print("Adding file {} to the data from json {}".format(name, os.path.basename(self.json_path)))
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                height, width = img.shape[:2]
                self._json_data["images"].append(self.entry_img(name,height,width, max_id))
        else:
            names = [os.path.basename(path) for path in list_of_paths]
            self.remove_by_name(names)

        self._internal_clearing()
    
    def clean_categories(self, save=False):
        """
        This method cleans the internal json data's categories.  It decides that
        if there is no annotations for a category it removes the category found from
        the internal json data.
        
        Parameters
        ----------
        save : bool, optional
             (Default value = False)
             This option is used to save the internal json data to the json file
             that was used to give the CVJ object it's data.
        
        Returns
        -------
        list : list
            The return value is named "remove_list" and it is returning a list of 
            categories that have been removed from the internal json data.
        """


        remove_list = []
        for class_, anns in self.get_class_id_2_anns().items():
            if len(anns) <= 0:
                remove_list.append(class_)

        for class_ in remove_list:
            self._json_data["categories"].remove(class_)
        
        if save:
            self.save_internal_json()

        self._internal_clearing()
        return remove_list
            
    def clean_images(self, save=False):
        """
        This method cleans the internal json data's images.  It decides that
        if there is no annotations for an image then it removes the image found from
        the internal json data.
        
        Parameters
        ----------
        save : bool, optional
             (Default value = False)
             This option is used to save the internal json data to the json file
             that was used to give the CVJ object it's data.
        
        Returns
        -------
        list : list
            The return value is named "remove_list" and it is returning a list of 
            image attributes that have been removed from the internal json data.
        """

        remove_list = []

        for img_id, anns in self.get_image_id_2_anns().items():
            if len(anns) <= 0:
                remove_list.append(self.get_image_id_2_image_attribs()[img_id])

        for obj in remove_list:
            self._json_data["images"].remove(obj)

        if save:
            self.save_internal_json()

        self._internal_clearing()
        return remove_list

    def remove_by_name(self, list_of_image_names, save=False):
        """
        This method removes all of the annotations and images associated with
        the list of image names supplied.  The image names must be the basenames of any
        file.  This method will clean the internal json data categories after completeing the 
        removal of images and the annotations associated with them.
        
        Parameters
        ----------
        list_of_image_names : list
             This argument is the list of basenames for the images to be removed.  So they must
             be names like "8.png, 8.tif, 4.jpeg" and not like "home/User/8.png, server/Desktop/5.tif".

        save : bool, optional
             (Default value = False)
             This option is used to save the internal json data to the json file
             that was used to give the CVJ object it's data.
        
        Returns
        -------
        list : list
            The first return value is named "list_of_image_names" which is just the list
            of names that was supplied.

        list : list
            The second return value is named "imgs" and it is returning a list of 
            image attributes that have been removed from the internal json data.

        list : list
            The third return value is named "anns" and it is returning a list of 
            annotations that have been removed from the internal json data.

        list : list
            The fourth return value is named "cats" and it is returning a list of 
            categories that have been removed from the internal json data. If those
            cateogories no longer have annotations associated with them.

        """
        imgs = []
        anns = []
        for name in list_of_image_names:
            for img in self._json_data["images"]:
                if img["file_name"] == name:
                    imgs.append(img)
            try:
                id = self.get_filename_2_image_id()[name]
            except KeyError:
                print("{} not found in internal json".format(name))
                continue

            for ann in self._json_data["annotations"]:
                if id == ann["image_id"]:
                    anns.append(ann)

        #print("Images found are below")
        #print(imgs)
        for img in imgs:
            print("removing {} from the JSON data loaded dfrom {}.".format(img["file_name"], os.path.basename(self.json_path)))
            self._json_data["images"].remove(img)

        for ann in anns:
            self._json_data["annotations"].remove(ann)

        cats = self.clean_categories()

        if save:
            self.save_internal_json()

        self._internal_clearing()
        return list_of_image_names, imgs, anns, cats


    def save_internal_json(self, save_name=None):
        """
        This method saves the internal json data dictionary.  This method is available
        when updates to the internal are done.  If the save_name variable is supplied it will
        be saved at that location with that name.  Else it will overwrite the json that was given 
        to the object.

        Parameters
        ----------
        save_name : string
             (Default value = None)
            This parameter is the name of the file.  While I am saying name, I mean it could be a
            file path plus the actual file name.

        Returns
        -------
        dict : dict
            * The internal json dictionary.
        """

        if save_name == None:
            print("Saving internal JSON data to the file {}.".format(os.path.basename(self.json_path)))
            with open(self.json_path, "w") as file:
                json.dump(self._json_data, file)
        else:
            print("Saving internal JSON data to the file {}.".format(os.path.basename(save_name)))
            with open(save_name, 'w') as file:
                json.dump(self._json_data, file)

        return self._json_data.copy()
        
    def _internal_clearing(self):
        """
        This method clears all of the data that was temporarily housed, just
        in case it was deep copied and there was a change in the internal json
        data.
        
        """

        self._class_to_filepath_data = None
        self._image_id_2_anns_dict = None
        self._class_id_2_name_dict = None
        self._class_name_2_id_dict = None
        self._img_id_2_filename_dict = None
        self._filename_2_image_id_dict = None
        self._imageid_to_filepath_dict = None
        self._image_id_2_image_attribs = None
        self._class_id_2_anns_dict = None

    def transfer_negatives_to_other_json(self,path_to_images=None, cvj_obj=None, json_data=None, json_path=None, save=False):
        """
        This method looks for negative sample type images in the internal json data
        that was created by the Painter class and then transfers those images to
        another json that is supplied via a path to a json file or the actual data from
        the json to be transferred to.

        Parameters
        ----------

        path_to_images : string
             (Default = None)
             This is the path that needs to be pointing to the negative images.  This is used to get
             the path names and check the images for height and width.  If an error occurs during the check
             it means that the file wasn't read correctly be opencv and your file may be corrupt.

        cvj_obj : CVJ
             (Default value = None)
             This argument is for a CVJ object that has already been loaded with a json path
             or json data.  If you need to know what the CVJ object is read "Introduction to the
             CVJ"

        json_data : dictionary
             (Default value = None)
             This value is used to transfer the negative images over to a dictionary that
             is already COCO formatted.  So if the user calling this method has loaded a json
             file already

        json_path : string
             (Default value = None)
             This is a path to a COCO formatted json file.  In this method it is used to create
             a CVJ object

        save : bool
             (Default value = None)
             This argument is used to save the internal json at the json path supplied to the object
             that has called this method.  It is defaulted to False becuase it could take a while to
             save.  This is up for the user to decide.
        
        Returns
        -------
        CVJ : CVJ
            The first return value is "cvj_obj" which is a CVJ object.  This holds
            the transferred images now and will need to be saved by the user.  If a json path
            was supplied here, upon returning the user can call teh "save_internal_json()" and
            it will save it where the json file is. If needing to understand what
            the CVJ object is refer to "Introduction to the CVJ".

        list : list
            The second return value is "imgs" which is a list of the images that have been
            transferred to the supplied json data.
        """

        if path_to_images == None:
            path_to_images = self.image_folder_path
            
        
        imgs = []
        
        for img in self._json_data["images"]:
            if img["file_name"].split('_')[0] == "negative":
                    imgs.append(img)

        for img in imgs:
            print("removing {} from the JSON data from {}.".format(img["file_name"], os.path.basename(self.json_path)))
            self._json_data["images"].remove(img)

        imgs = [os.path.join(path_to_images,img["file_name"]) for img in imgs]

        if json_data != None:
            cvj_obj = CVJ()
            cvj_obj._json_data = json_data
            cvj_obj.update_images(imgs)

        elif json_path != None:
            cvj_obj = CVJ(json_path=json_path)
            cvj_obj.update_images(imgs)

        elif cvj_obj != None:
            cvj_obj.update_images(imgs)

        if save:
            self.save_internal_json()
        
        self._internal_clearing()
        return cvj_obj, imgs

    def replace_extensions_of_json_images(self, replacement=".png", save=False):
        """
        This method replaces the file extension of the images to the replacement type given.
        
        Parameters
        ----------
        replacement : string
            (Default = .png)
             This is the variable to replace the extensions with.
            
        save : bool
            (Default = False)
            This is used to save the internal json data to the json file found at the path given using cvj_obj.json_path = "path/to/your/json"
        
        Returns
        -------
        CVJ : CVJ
            This returns the CVJ object.
        """

        images = self._json_data["images"]

        print("Replacing extensions within the internal json data.")
        for image in images:
            filename = image["file_name"]
            newname = filename.split('.')[0] + replacement
            image["file_name"] = newname

        if save:
            self.save_internal_json()

        return self


    def entry_bbox(self, bbox, class_id, image_id, id):
        """
        This method assists with entering valid annotations.

        Parameters
        ----------
        bbox : list
            This parameter is bounding box coordinates in the format of [x, y, width, height]

        class_id : int
            This parameter is the class id also known as the category id

        image_id : int
            This parameter is the image id that this annotation belongs to 

        id : int
            This parameter is the id of the annotation

        Returns
        -------
        dict : dict
            * keys   = annotation format found in "Introduction to the CVJ"

            * values = the values supplied.

        """
        assert len(bbox) == 4, str(bbox)
        assert isinstance(class_id,int), str(type(class_id))

        # hard to validate bbox format numerically,
        # just make sure it is: (col0, row0, width, height)

        bbox = list([int(vv) for vv in bbox])
        # assert bbox[2] > 0 and bbox[3] > 0, str(bbox)

        return {'area':        int(bbox[2]*bbox[3]),
                'bbox':        bbox,
                'category_id': int(class_id),
                'id':          int(id),
                'image_id':    int(image_id),
                'iscrowd':     0,
                # each segmentation polygon: [x0, y0, x1, y1, ..., xn, yn]
                'segmentation': [[bbox[0],        bbox[1],         # topleft
                                bbox[0]+bbox[2],bbox[1],         # topright
                                bbox[0]+bbox[2],bbox[1]+bbox[3], # lowerright
                                bbox[0],        bbox[1]+bbox[3], # lowerleft
                                ]],
                }

    def create_empty_json(self):
        """
        This method assists in making the json for a COCO format.

        Parameters
        ----------

        Returns
        -------
        dict : dict

            * keys   = "images", "categories", "annotations". Example found in "Introduction to the CVJ"

        """
        return {'images':     [],
                'categories': [],
                'annotations':[],
                }

    def entry_img(self, file_name, height, width, id):
        """
        This method assists with entering an image and the attributes of
        that image.

        Parameters
        ----------
        file_name : string
            This parameter is the filename of the image that is being inserted in to the json file

        height : int
            This parameter is the height of the image. Normally this is equivalent to img.shape[0] when using numpy

        width : int
            This parameter is the width of the image. Normally this is equivalent to img.shape[1] when using numpy

        id : int
            This parameter is the image id and file name are the most crucial components to this entry.
            Without the image_id most of the functions in this object will not help the user)

        Returns
        -------
        dict : dict

            * keys   = "file_name", "height", "width", "id".  For more explanation see "Introduction to the CVJ"
        """
        assert isinstance(file_name,str), str(type(file_name))
        assert isinstance(height,int), str(type(height))
        assert isinstance(width, int), str(type(width))
        assert isinstance(id,   int), str(type(id))
        ret = {'file_name': file_name,
                'height': int(height),
                'width':  int(width),
                'id':     int(id)}

        return ret

    def categ_idx_to_coco_categ(self, id):
        """
        This method creates a pseudo COCO category annotation.
        This type of annotation is mostly useless besides the actual class id.
        AKA category id.

        Parameters
        ----------
        id : int
            This parameter is the class id, AKA the category id

        Returns
        -------
        dict : dict

            * keys   = "id", "name", "supercategory".  For further explanation see "Introduction to the CVJ"
        """
        assert isinstance(id,int), str(type(id))
        id = int(id)
        
        if id == 63428483 or id == str(63428483): # this is the negative class id created by CVJSON
           return {'id':id, 'name':str("NEGATIVE"), 'supercategory':str(id)}
        
        return {'id':id, 'name':str(self.get_class_id_2_name()[id]), 'supercategory':str(id)}

    def new_category(self, class_id, name, super_category_id = None):
        """
        This method creates a new category in the format of COCO.

        Parameters
        ----------
        id : int
            This parameter is the class id, AKA the category id
        name : str
            This parameter is the actual name of the category
        super_category_id: str
            (Default = None)
            This parameter is the category that is above this one if using a
            hierarchy.  Otherwise this will just be the string version of the id

        Returns
        -------
        dict : dict

            * keys   = "id", "name", "supercategory".  For further explanation see "Introduction to the CVJ"
        """
        
        class_id = int(class_id)
        name = str(name)
        if super_category_id == None:
            super_category_id = str(class_id)

        if id == 63428483 or id == str(63428483): # this is the negative class id created by CVJSON
           return {'id':id, 'name':"NEGATIVE", 'supercategory':super_category_id}

        return {'id':class_id, 'name':name, 'supercategory':super_category_id}




