"""
Author: Benjamin Anderson Garrard

This script houses functions that transform images
and the bounding boxes associated with them.

"""
import os,sys
import sys

import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from ..cvj import CVJ 
  
class Augmenter(CVJ): 
    """
    The Augment class is used to augment images using
    the imgaug library.  just pip install imgaug if you
    do not have it.

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

    def transf_one_patch(self,patch, bboxes, PCA_Augment=True, edge_enhancement=False, color_jittering=True, preview=False, preview_boxes=False):
        """
            
        This method is used to transform just one image at a time,
        this method also transforms the bounding boxes accordingly.

        Rotation and flipping is not handled within this function. #TODO: create a method that does the rotation and flipping.
        The reason for this is that the bounding boxes that are transformed become too
        large for small object detection.

        https://arxiv.org/pdf/1708.06020.pdf is a paper explaining why these augmentations
        were chosen.    
        
        Parameters
        ----------
        patch : numpy array
             The image to augment

        bboxes : numpy array
             A numpy array of the bounding boxes associated with the supplied image/patch

        PCA_Augment : bool
             (Default value = True)
             Is a boolean that augments the image based on PCA color augmentation.
             PCA color augmentation just means taking scaled intensities of the most import 
             colors and multiplying them through the image.

        edge_enhancement : bool
             (Default value = False)
             Is a boolean that starts a convolution that enhances the edges in the image and makes them more
             prominent.  This could make classifier find edges more quickly and then find more
             complex features associated with the edges in later convolutions if using 
             deep learning.

        color_jittering : bool
             (Default value = True)
             This is a boolean to start adding pixel values ranging from -20 to 20 to RGB channels randomly

        preview : bool
             (Default value = False)
             Is a boolean that starts up a comparison from the image/patch supplied to the new augmented one.

        preview_boxes : bool
             (Default value = False)
             Is a boolean that allows the boxes to be drawn on the images in the preview.  The original and 
             transformed versions.

        Returns
        -------
        numpy array : The first numpy array is the image/patch
        numpy array : The second numpy is the bounding boxes associated with the image.
        """

        
        bboxes = self.xywh_to_xyxy(bboxes)

        ##########################################
        # Geometric augmentations
        # These are the most important ones
        # for getting better classifications
        ##########################################
        patch, bboxes, previous_patch, preview_boxes = self.augment_geo(patch, bboxes)
        

        #######################################
        # Photometric augmentations below
        #######################################
        if PCA_Augment:
            patch = self.fancy_pca(patch, alpha_std=.1)
            #patch = PCA_augmentation(patch)

        if edge_enhancement:
            edge_enhancement_filter = np.array([[ -1, -1, -1],
                                                [-1, 10, -1],
                                                [-1, -1, -1]])
            aug = iaa.Convolve(matrix=edge_enhancement_filter)
            patch = aug.augment_image(patch)

        if color_jittering:
            aug = iaa.Add((-20, 20), per_channel=0.5)

            patch = aug.augment_image(patch)

        if preview:
        # Just for viewing the work
            if preview_boxes:
                for bbox in bboxes:
                    cv2.rectangle(patch, (bbox[0], bbox[1]), (bbox[2],bbox[3]), color= (255,0,0), thickness=1)

                for bbox in preview_boxes:
                    cv2.rectangle(previous_patch, (bbox[0], bbox[1]), (bbox[2],bbox[3]), color= (255,0,255), thickness=1)

            cv2.imshow("before_augment", previous_patch)
            cv2.imshow("augmented", patch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        bboxes = self.xyxy_to_xywh(bboxes)
        return patch, bboxes



    def augment_geo(self,patch, bboxes):
        """ 
        This method uses the imaug library 
        https://github.com/aleju/imgaug
        to implement all augmentations since
        it can handle bounding boxes also.

        Rotation seems to rescale the bounding boxes too much
        so the rotation and flips should be handled by Detectron and not
        by this method
        

        Parameters
        ----------
        patch : numpy array
            This numpy array is the image.  Opencv, skimage, or numpy can generate this.
            
        bboxes : numpy array
            This numpy array holds the bounding boxes that are associated with this image.  They are needed to be in the format
            of [[x1, y1, x2, y2]
            

        Returns
        -------
        numpy array : The first parameter "img_aug" returned is the numpy array for the image.

        numpy array : The second parameter "bbox_" returned is the numpy array for the bounding boxes that have been
                      been transformed with the image returned in the first parameter.

        numpy array : The third parameter "patch" is the numpy array for the image before it was altered.

        numpy array : The fourth parameter "previous_boxes" returned is the numpy array for the bounding boxes that have 
                      not been altered associated with the returned third parameter.


        """

        bbox_list = []
        for bbox in bboxes:
            bbox_list.append(ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
            
        bbs = ia.BoundingBoxesOnImage(bbox_list, shape=patch.shape)

        '''
        When scaling just upscale and not downscale
        '''
        seq = iaa.Sequential([
            iaa.Multiply((1.1, 1.7)), # change brightness, doesn't affect BBs
            iaa.Affine(
                translate_px={"x": 10, "y": 10},
                scale=(1.05, 2),
                mode="reflect",
                shear=(2, 5),
                order=ia.ALL
            ), # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            #iaa.EdgeDetect(alpha=.1)
            
        ])

        
        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will
        # lead to the same augmentations.
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
        # exactly same augmentations for every batch!
        seq_det = seq.to_deterministic()

        # Augment BBs and images.
        # As we only have one image and list of BBs, we use
        # [image] and [bbs] to turn both into lists (batches) for the
        # functions and then [0] to reverse that. In a real experiment, your
        # variables would likely already be lists.
        image_aug = seq_det.augment_images([patch])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        '''
        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
            )

        # image with BBs before/after augmentation (shown below)
        image_before = bbs.draw_on_image(patch, thickness=1)
        
        image_after = bbs_aug.draw_on_image(image_aug, thickness=1, color=[0, 0, 255])
        '''
        
        previous_boxes = []
        for bbox in bbs.bounding_boxes:
            previous_boxes.append([bbox.x1,bbox.y1, bbox.x2, bbox.y2])

        bbox_ = []
        for bbox in bbs_aug.bounding_boxes:
            bbox_.append([bbox.x1,bbox.y1, bbox.x2, bbox.y2])

        bbox_ = np.asarray(bbox_)

        previous_boxes = np.asarray(previous_boxes)

        return image_aug, bbox_, patch, previous_boxes






    def fancy_pca(self, img, alpha_std=0.1):

        """
        Fancy PCA is a photometric data augmentation technique that finds the most
        important color values, 3 because of the 3 channels, then gets a scaled intensity of
        the colors and augments with those intensity * a random variable.

        Parameters
        ----------
        img : numpy array with (h, w, rgb) shape, as ints between 0-255)
            
        alpha_std : how much to perturb/scale the eigen vecs and vals the paper used std=0.1
             (Default value = 0.1)

        Returns
        -------
        numpy array : image-like array as float range(0, 1)

        NOTE: Depending on what is originating the image data and what is receiving
        the image data returning the values in the expected form is very important
        in having this work correctly. If you receive the image values as UINT 0-255
        then it's probably best to return in the same format. (As this
        implementation does). If the image comes in as float values ranging from
        0.0 to 1.0 then this function should be modified to return the same.
        Otherwise this can lead to very frustrating and difficult to troubleshoot
        problems in the image processing pipeline.
        This is 'Fancy PCA' from:
        # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        #######################
        #### FROM THE PAPER ###
        #######################
        "The second form of data augmentation consists of altering the intensities
        of the RGB channels in training images. Specifically, we perform PCA on the
        set of RGB pixel values throughout the ImageNet training set. To each
        training image, we add multiples of the found principal components, with
        magnitudes proportional to the corresponding eigenvalues times a random
        variable drawn from a Gaussian with mean zero and standard deviation 0.1.
        Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
        we add the following quantity:
        [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
        Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
        matrix of RGB pixel values, respectively, and αi is the aforementioned
        random variable. Each αi is drawn only once for all the pixels of a
        particular training image until that image is used for training again, at
        which point it is re-drawn. This scheme approximately captures an important
        property of natural images, namely, that object identity is invariant to
        change."
        ### END ###############
        Other useful resources for getting this working:
        # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
        # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
        """


        orig_img = img.astype(float).copy()

        img = img / 255.0  # rescale to 0 to 1 range

        # flatten image to columns of RGB
        img_rs = img.reshape(-1, 3)
        # img_rs shape (640000, 3)

        # center mean
        img_centered = img_rs - np.mean(img_rs, axis=0)

        # paper says 3x3 covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        #     eig_vals [0.00154689 0.00448816 0.18438678]

        #     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
        #      [-0.81323938  0.05207541 -0.57959456]
        #      [ 0.45878547  0.67008619 -0.58352411]]

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        # get [p1, p2, p3]
        m1 = np.column_stack((eig_vecs))

        # get 3x1 matrix of eigen values multiplied by random variable draw from normal
        # distribution with mean of 0 and standard deviation of 0.1
        m2 = np.zeros((3, 1))
        # according to the paper alpha should only be draw once per augmentation (not once per channel)
        alpha = np.random.normal(0, alpha_std)

        # broad cast to speed things up
        m2[:, 0] = alpha * eig_vals[:]

        # this is the vector that we're going to add to each pixel in a moment
        add_vect = np.matrix(m1) * np.matrix(m2)

        for idx in range(3):   # RGB
            orig_img[..., idx] += add_vect[idx]

        # for image processing it was found that working with float 0.0 to 1.0
        # was easier than integers between 0-255
        # orig_img /= 255.0
        orig_img = np.clip(orig_img, 0.0, 255.0)

        # orig_img *= 255
        orig_img = orig_img.astype(np.uint8)

        # about 100x faster after vectorizing the numpy, it will be even faster later
        # since currently it's working on full size images and not small, square
        # images that will be fed in later as part of the post processing before being
        # sent into the model
        # print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

        return orig_img


if __name__ == "__main__":
    print("do not run this script as the main application.  This is an extension to the CVJ object")