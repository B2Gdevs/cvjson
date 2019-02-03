***************************
Introduction to the Cropper
***************************

The real beauty of the library is the extension ability.  In this case
we are going over the Cropper and how it will actually manipulate the images
supplied along side the JSON file that was supplied.  


Basic Setup
-----------

.. code-block:: python

    from cvjson.extensions.cropper import Cropper

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    
    cvj_obj = CVJ(json_path=json_path)
    cropper = Cropper(cvj_obj)

    cropper.image_folder_path = image_folder_path


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

    from cvjson.extensions.Cropper import Cropper

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    save_directory    = "path/to/store/images"
    
    cvj_obj = CVJ(json_path=json_path)
    cropper = Cropper(cvj_obj)

    cropper.image_folder_path = image_folder_path

    cropper.crop_images_boxes_centered(save_directory, mininum_image_size = 300, maximum_image_size = 500)

Booyah, with 7 lines of code we are able to crop images to the center of bounding boxes and store them elsewhere while maintaining the bounding 
boxes new positions.  A little caveat to this function that at a minimum there will be as many images as there are bounding boxes within the JSON
that is supplied to the CVJ object.

...ok that last part technically isn't true.  You could set the testing variable in the method and it will only crop from that many images in order.

For example:
    
    This code will crop from images starting from image_id 1 up to 2 and crop from there and output the JSON file needed.  

    .. code-block:: python
         
         # The testing variable is now set
        cropper.crop_images_boxes_centered(save_directory, mininum_image_size = 300, maximum_image_size = 500, testing=2)

This variable was made with just testing in mind.  However, if you really needed just a random subset of the images with centered objects then 
setting that testing variable is a good idea.


Oversample and undersample
**************************

The crop_images_boxes_centered() method actually can oversample rare classes and undersample classes that are too
prominent in the data.  Keep in mind I am talking about oversampling and undersampling the IMAGES that are cropped
using the method.  This will not undersample or oversample bounding boxes because that wouldn't make sense here.  Not unless
someone could maximize crops that begin to even out bounding box annotations.  However, the crop_images_boxes_centered() method does
not do that.

To actually oversample and undersample, all one has to do is set the image threshold to how many images they want each class to have.
This is directly related to which object is in the center of the image.  An image belongs to a cdrtain class if the object in the center is that
certain class.

.. code-block:: python

    cropper.crop_images_boxes_centered(save_directory, mininum_image_size = 300, maximum_image_size = 500, image_threshold=100)

The code above will run through the JSON data, crop images based on th bounding boxes.  The crops will range from 300 to 500 pixel sizes.  
Then the code will find 100 images belonging to each class it found and stop at that if there are enough images.  Otherwise, it begins
to use the Augmenter class and generates new images up to the image threshold.

The Other Variables
*******************

    .. code-block:: python

        cropper.crop_images_bbox_centered(save_directory, max_image_size = 768 , min_image_size = 128, max_bounding_box_size=600, scale = 1, padding = 0,padded_beforehand=False, image_threshold=0, cv2enum=cv2.BORDER_REFLECT, testing=0):

The **padded_beforehand** variable tells the method that the image was padded before this method was called. If this is false then the method
will pad image supplied by the amount of padding that is supplied.  However, the bounding boxes will not be tracked.  The best 
use for this is just to use the Painter class to pad, track, and inpaint. Then use the cropper, keep this variable True and 
the padding match the padding used in the Painter class.

The **cv2enum** is the padding method to use in opencv so someone can use cv2.BORDER_REFLECT for a reflected border around the images and the cropper will crop 
using that method and how much padding was supplied.  However, this does not keep track of the bounding boxes in the padded regions if that is wanted.  
The Painter class has methods that can find them, but that is used for inpainting and does nto return those values if needed.  To make that ability for 
this class or the Painter class, one will have to dig through the code and modify it.

The **scale** variable is used to create a crop that is used to crop an image to at least that scale of the bounding box.  So if the scale was 2 then someone
would get a minimum crop size of the bounding box diameter * 2.  This is the minimum only if it is within the range for min_image_size and max_image_size.

The **max_bounding_box_size** is set to remove any images trying to be created from a bounding box size that is bigger than the max_bounding_box_size that is
set.


Trap
----

Using the cropper to generate large images could possibly overload the RAM.  This is because the crops are handle all in one go for each image.  So if a particular
image has a lot of annotations, you will store that same amount of images in the buffer until they are annotated and saved.  Either the image size needs
to be in mind when cropping, the RAM, or the images trying to be cropped from.


Road Map of Features
--------------------

1. Allow the user to specify how the oversampling data augmentation is done.  (Example: GANS, imgaug, photometric-only, geometric-only, etc)
