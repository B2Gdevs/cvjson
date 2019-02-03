***********************
Introduction to the CVJ
***********************

This is the beginning documentation of a
python project known as CVJSON.  The project
is basically an extension module for python dictionaries
that are handling COCO formatted data.

Why Use This Library
--------------------

There are a significant amount of reasons for using the library.
However I am only going to talk about the 2 main ones.  Reducing the redundant
utility scripting and this is a way to have everyone benefit from each others 
utility programming. 

If you needed more functionality and wrote a piece of code for it, add it to this
library and then everyone else can benefit also.  


.. _COCO-Format:

COCO Format Needed Don't Skip This Section
------------------------------------------

The COCO format has different formats for different types of detections. We 
are looking at the object detection format for most of
this library's functionality.  Even though this library supports
COCO, it is a partial support since the tracking is only done for portions
of the format.  The next section describes how to format the json

The basic template portion of COCO is like this

.. code-block:: python

    {
    "info" : info,
    "images" : [image],
    "annotations" : [annotation],
    "licenses" : [license],
    }

Then from here there is further formatting. An "annotation"
needs to be in the below format which is part of the "annotations" list 
in the above example. Then categories list needs to be 
added to the above format.

.. code-block:: python

    From COCO http://cocodataset.org/#format-data

    annotation  : {
                  "id": int,
                  "image_id": int,
                  "category_id": int,
                  "segmentation": RLE or [polygon],
                  "area": float,
                  "bbox": [x,y,width,height],
                  "iscrowd": 0 or 1,
                  }

    categories  : [{
                    "id": int,
                    "name": str,
                    "supercategory": str,
                     }],

.. _main structure:

Main Structure
--------------

Which then leaves us with the main structure below. Since we have just
added the "categories" list to the format.

.. code-block:: python

    {
    "info" : info,
    "images" : [image],
    "annotations" : [annotation],
    "licenses" : [license],
    "categories": [category]
    }

However, this library is pretty much just using and 
checking for the part of the json that is like below,
if it is like the one above, that is fine, the library 
should be able to handle it.

.. code-block:: python

    {
    "images" : [image],
    "annotations" : [annotation],
    "categories": [category]
    }

Remember though that the json needs to have the annotation's in the 
format like below.

.. code-block:: python

    annotation  : {
                  "id": int,
                  "image_id": int,
                  "category_id": int,
                  "segmentation": RLE or [polygon],
                  "area": float,
                  "bbox": [x,y,width,height],
                  "iscrowd": 0 or 1,
                  }

Also each image annotation must have at least "id", "width", "height", and "file_name"
from the structure below.  Think that every image annotation that is in the "images" list
that is a part of the :ref:`main structure`

.. code-block:: python

    image: {
            "id" : int,
            "width" : int,
            "height" : int,
            "file_name" : str,
            "license" : int,
            "flickr_url" : str,
            "coco_url" : str,
            "date_captured" : datetime,
        }


The library does not support the "license", "flickr_url", "coco_url",
or "date_captured" keys.  If creating a json with this library or adding annotations it will be in the format 
below for image annoations.

.. code-block:: python

    image: {
            "id" : int,
            "width" : int,
            "height" : int,
            "file_name" : str
        }

Quick Start
-----------

Go to `here <https://bitbucket.org/mayachitrainc/cvjson/src/master/>`_ and download the git 
repo. Afterwards just change directory to the newly downloaded cvjson repo and type "pip install".

So a walk through of those steps would be like below:

.. code-block:: bash

    git clone https://bitbucket.org/mayachitrainc/cvjson/src/master/
    pip install cvjson/

Now if you plan to use the Painter extension then there is another program to install.  That program is GMIC,
which is a powerful plugin used in the GIMP editing software.  

.. code-block:: bash
    
    sudo apt-get install gmic


Now for further details go to :ref:`The-Basics` to dive right in.

.. _The-Basics:

The Basics
----------

The basic object is a CVJ (Computer Vision JSON) which holds the json data, path, etc.

Below is how to get the average area of each class and plot it
using seaborn.  With three lines of code you now have more insights on 
object data.

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    cvj_object = CVJ(json_path)
    classes = cvj_object.get_average_area_by_class(show_plot=True)

An example of how one can get the class distribution would be:

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "/home/typhon/Downloads/full_xview_coco_.json" 

    cvj_obj = CVJ(json_path=json_path)
    cvj_obj.get_distribution_of_class_id(show_plot=True)

The above example returns a list of class id's that
are not unique.  But the real part is that this will
use seaborn in the backend and generate a histogram
based on that list.  The list is just returned in
case you want to plot it differently.

CVJ object is a Dictionary!?
*****************************

A cool feature is that you can use this object just
like a normal dictionary that you loaded your json file from.

.. code-block:: python
    
    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    cvj_object = CVJ(json_path)
    annotations = cvj_object["annotations"]


All of the functions you would find in a dictionary is included
with the object.  Mainly because it is using a dictionary to hold the
information and not just the __dict__ that represents the object.
If needing to understand what a __dict__ is for an python object, `here <https://stackoverflow.com/questions/19907442/python-explain-dict-attribute/>`_
is a link that gives a good idea.

You can also iterate through the object just like you would
if you loaded the json in to a python dictionary.

Iterating the CVJ Object!?
*****************************

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    cvj_obj = CVJ(json_path)

    for key in cvj_object:
        print(key)

The output for that should be "images", "categories", and "annoations" if the
supplied json is correct, which it should be since there are checks in place to
see if it is a COCO formatted json.  For now only COCO is supported, but the idea
to convert other JSON formats dealing with computer vision have been an idea.  
Most likely if that were to be supported it would take in the other JSON type and
convert it to COCO and you would interact with it like a COCO formatted dictionary.

Basic Dictionary Subsets of the JSON Data
*****************************************

So the CVJ object creates dictionaries to quickly access data and
a lot of the time these dictionaries will need to be used. 

Below is a coded example of all of the dictionaries that are
regularly used and how to get them.  

.. code-block:: python

    """ 
    These aren't real enums, but class variables used like enums

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

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    cvj_object = CVJ(json_path)

    image_id_2_anns = cvj_obj.get_dictionary(CVJ.IMID_2_ANNS)
    class_id_2_class_name = cvj_obj.get_dictionary(CVJ.CLID_2_NAME)
    class_name_2_class_id = cvj_obj.get_dictionary(CVJ.CLNAME_2_CLID)
    image_id_2_file_name = cvj_obj.get_dictionary(CVJ.IMID_2_FNAME)
    file_name_2_image_id = cvj_obj.get_dictionary(CVJ.FNAME_2_IMID)
    image_id_2_file_path = cvj_obj.get_dictionary(CVJ.IMID_2_FPATH)
    image_id_2_image_attributes = cvj_obj.get_dictionary(CVJ.IMID_2_IMATTR)
    class_id_2_annotations = cvj_obj.get_dictionary(CVJ.CLID_2_ANNS)

One can use the getter methods that are similar to these dictionaries and grab the info like a normal, but
the above dictionaries can also be used just in case.

Basic Data Insights
*******************

Each of the below are basic data insight methods that by defualt
plot information out using seaborn in the backend.  Well, once you have 
loaded a correctly formatted JSON as described at the first section 
found here -> :ref:`COCO-Format`

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    cvj_object = CVJ(json_path)

    cvj_obj.get_class_count_by_image_id(some_image_id)
    cvj_object.get_distribution_of_class_id()
    cvj_object.get_distribution_of_area()
    cvj_object.get_average_area_by_class()
    cvj_object.get_bbox_count_by_class_dict()
    cvj_object.get_average_side_lengths()



The Convenience and the Traps
-----------------------------

The getter methods can return many types of values depending on what get
method is called.  The first example is the normal way one would use a 
getter method. 

Example 1
*********

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"
    # The image ID is arbitrary in this example
    img_id = 1

    # This grabs the annotations associated with that image id
    annoations_for_img_id_one = cvj_obj.get_image_id_2_anns(img_id)

Example 1 shows the normal use of the get method.  The user puts in the image ID
for which the method returns the annoations for that particular image.

The return will change depending on what is or is not given.  The example below shows 
that the user will get the dictionary that is used to find the annotations in **Example 1** 
if there are no arguments given.

Example 2
*********

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    # The image ID is arbitrary in this example
    img_id = 1

    cvj_obj = CVJ(json_path)

    # This grabs the dictionary from image ID to annoations from the data that was gathered
    # from the JSON file loaded in to the object during instantiation
    image_id_2_anns_dict= cvj_obj.get_image_id_2_anns()

So since the user now has the dictionary from the method they can use it like the following.

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "the/path/to/your/json"

    # The image ID is arbitrary in this example
    img_id = 1

    cvj_obj = CVJ(json_path)

    # This grabs the dictionary from image ID to annoations from the data that was gathered
    # from the JSON file loaded in to the object during instantiation
    image_id_2_anns_dict= cvj_obj.get_image_id_2_anns()

    annotions_for_image_id_1 = image_id_2_anns_dict[img_id]

Now the functionality of the same method can change even further on if some other json data that is supplied
to the method.  Sometimes a need arises where one has manipulated the data in a new json-type dictionary
and needs information from that dictionary.  This functionality is used within the Cropper extension/subclass.

**RULE : If you supply this method with JSON data using the json_data variable, everything returned will always be from 
the newly supplied data**

Example 3
*********

.. code-block:: python

    from cvjson.cvj import CVJ
    import json

    json_path = "the/path/to/your/json"
    some_other_json = "path/to/another/json"

    with open(some_other_json, 'r') as file:
        other_json_data = json.load(file)

    # The image ID is arbitrary in this example
    img_id = 1

    cvj_obj = CVJ(json_path)

    # This grabs the annotations associated with that image id from the supplied json data
    annoations_for_img_id_one_from_other_json_data = cvj_obj.get_image_id_2_anns(img_id=img_id, json_data=other_json_data)

What this is doing now is grabbing the data from the variable "other_json_data" and is searching for the annotations that
are associated with the image id also supplied.  By default this is not how the method works obviously because it is not intuitive,
but the need does arise where this functionality is preferable.

Below shows that the user will get the dictionary that has keys as image IDs and values of annotations **from the json data supplied to the
json_data variable**.

Example 4
*********

.. code-block:: python

    from cvjson.cvj import CVJ
    import json

    json_path = "the/path/to/your/json"
    some_other_json = "path/to/another/json"

    with open(some_other_json, 'r') as file:
        other_json_data = json.load(file)

    # The image ID is arbitrary in this example
    img_id = 1

    cvj_obj = CVJ(json_path)

    # This grabs the dictionary from image ID to annoations for the supplied json data
    image_id_2_anns_dict_for_some_other_json = cvj_obj.get_image_id_2_anns(json_data=other_json_data)

So the above got the dictionary for the json_data that was supplied to the method rather than the json_data that
is within the CVJ object.


Now that that is settled there are more dictionary gathering methods that just 
that previous one.  These are all used to quickly find the data in the JSON that
was supplied.

.. code-block:: python

    from cvjson.cvj import CVJ

    json_path = "/home/typhon/Downloads/full_xview_coco_.json" 

    cvj_object = CVJ(json_path)

    """
    These get methods below are not used in the traditional sense right now. 
    They are returning dictionaries and do not require arguments.

    That is the Trap in this library
    """
    file_name_2_img_id_dict = cvj_obj.get_filename_2_image_id()
    image_id_2_filename_dict = cvj_obj.get_image_id_2_filename()
    image_id_2_annotations_dict = cvj_obj.get_image_id_2_anns()
    image_id_2_image_attributes_dict = cvj_obj.get_image_id_2_image_attribs()
    
    class_name_2_id_dict = cvj_obj.get_class_name_2_id()
    class_id_2_name_dict = cvj_obj.get_class_id_2_name()

    image_folder_path = "/home/typhon/Downloads/full_view_test"

    # To use get_image_id_2_filepath() the image folder path variable
    # must be set to the images associated with the JSON that has been supplied to the 
    # CVJ object.
    cvj_obj.image_folder_path = image_folder_path
    image_id_2_filepath_dict = cvj_obj.get_image_id_2_filepath()

    # this one is not a dictionary that gets returned it is a list
    image_ids = cvj_obj.get_image_ids()
