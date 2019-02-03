*****************************
Introduction to the Augmenter
*****************************

Welcome to the introduction of the Augmenter extension/class. This class
is used to handle all of the augmentation that should happen with the supplied
image folder path. This class, for now, is fully powered by OpenCV and the imgaug
library.

Basic Setup
-----------

.. code-block:: python

    from cvjson.extensions.augmenter import Augmenter

    json_path         = "path/to/your/json"
    image_folder_path = "path/to/your/images"
    
    cvj_obj = CVJ(json_path=json_path)
    augmenter = Augmenter(cvj_obj)

    augmenter.image_folder_path = image_folder_path


The Main Method
***************

The main method is transf_one_patch().  This method gives the images there photometric augmentations and the geometric augmentations.  
From the code below it can be seen that the required arguments is a patch/image and the bounding boxes for that patch/image.

.. code-block:: python


    augmenter.image_folder_path = image_folder_path
    augmenter.transf_one_patch(patch, bboxes, PCA_Augment=True, edge_enhancement=False, color_jittering=True, preview=False, preview_boxes=False)


The **preview** variable tells the method to do a before and after on the images and having the pop up using OpenCV.

The **preview_boxes** variable tells the method to show the bounding boxes during the preview.

Augments
--------

This `paper <https://arxiv.org/pdf/1708.06020.pdf/>`_ is what was referenced to find what types of augmentations would be useful.

The **PCA_Augment** variable is a boolean that tells the method to use a photometric augmentation known as "Fancy PCA".  The code documentation covers
it heavily but the paper can also be found `here <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf/>`_
to understand it more.


The **edge_enhancement** tells the method to use convolution to enhance the edges and make them more prominent. 

The **color_jittering** variable tells the method to add colors randomly through out the image.  Tweaking the actual code of this and finding the right parameters
of how much jittering is best might be worth looking in to.  


Road Map of Features
--------------------

1. Optimized color_jittering parameters for various machine learning that can be loaded.
2. GANS training
3. Pretrained GANS for out of the box use.

