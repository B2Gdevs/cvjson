***************************
Introduction to the Painter
***************************

Welcome to the Painter extension/class.  This class handles finding the bounding boxes
in an image, including the reflected ones, masks them and inpaints them using `GMIC <https://gmic.eu/which/>`_
is a plugin that is used in the GIMP editing software. 

Installation of GMIC
--------------------

GMIC has its own interpreter and needs to be installed.  Thankfully it is just the one command below.

.. code-block:: bash

    sudo apt-get install gmic


Basic Setup
-----------

Setup the objec to associate json with the images.

.. code-block:: python

    from cvjson.extensions.painter import Painter

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    
    cvj_obj = CVJ(json_path=json_path)
    augmenter = Painter(cvj_obj)

    painter.image_folder_path = image_folder_path

and now we are ready to move on to the main method of this class as of right now.

Main Methods
************

The generate_negatives() method is used to completely inpaint the image and not preserve the original image
inside of the newly inpainted image.

The generate_positives() method is used to completely inpaint the image and preserve the original image
inside of the newly inpainted image.  So it just inserts the original images in to the new inpainted images.

.. code-block:: python

    painter.generate_negatives( save_directory=None, n_cores=1, padding=0, generation_method=INPAINT)
    painter.generate_positives( save_directory=None, n_cores=1, padding=0, generation_method=INPAINT)

Both of these methods create the images in a temp folder inside the save_directory given and then when the method is finished running
it moves the images to the save save directory and then updates the annotations.  They both have multiprocessing capabilities and need to have
the generation method told to it.


Other Variables
***************

The **generation method** is set in place for people who build on this extension.  If someone creates a new generation method for painting in the image,
then they should make that method in the Painter class, set a class variable that is used to for generate_negatives and generate_positives to associate 
with the method.  The "INPAINT" argument that is passed is a class variable which is treated like an enum for the Painter class.  "INPAINT" tells the 
method to use the multi-patchbased methods to generate the images.
