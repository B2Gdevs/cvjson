***********************************
Introduction to the Image Organizer
***********************************

The real beauty of the library is the extension objects.  Each object can inherit
from each other.  If you want to extend your handle then just import the extension 
you want.


Example
*******

.. code-block:: python

    from {package_name}.exentsions.Cropper import Cropper

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    
    json_handle = Handle(json_path)
    cropper = Cropper(json_handle, image_folder_path)


Now that the cropper is instantiated we can crop images associated
with files.  In the below example we crop all of the images to 
bounding box centers, but we defined a minimum size and a maximum size
that means they will be at least the minimum and no greater than the maximum.
There is a count of how many images each class will have during this process.
An image threshold is used to stratify that count.  So if it was at 1000
then all classes will have 1000 images associated with them in the json.

There are more details in the code.

The image_threshold is 0 because if it were anything else then we would undersample
any class that is higher than the image threshold and oversample, using data augmentation, any class that is 
under the image_threshold. Right now I just want crops for each bounding box I have in the json file.

.. code-block:: python

    from {package_name}.exentsions.Cropper import Cropper

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    
    json_handle = Handle(json_path)
    cropper = Cropper(json_handle, image_folder_path)

    cropper.crop_images_boxes_centered(mininum_image_size = 300, maximum_image_size = 500, image_threshold=0)

Booyah, with 7 lines of code we are able to crop, augment, oversample, undersample, and stratify images and produce
a new json file that can be used for training.  Now your thinking, "where are the files?", well if the save directory
is not supplied then it pulls the image_folder_path directory and saves the images in that parent folder under {TIMESTAMP}_crop_default.