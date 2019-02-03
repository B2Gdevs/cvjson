'''''''''''''''''''''''''''''''''
Primary Author: Benjamin Garrard'
'''''''''''''''''''''''''''''''''

import cv2
import json
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
import subprocess
import glob
from .augmenter import Augmenter   # TODO: make this work with the Augment Class
import copy

from ..cvj import CVJ   
import os
from tqdm import tqdm
import datetime
 

class Cropper(CVJ):
    """
    This class is an extension to the CVJ object.  This classes sole purpose is to crop
    images and produce new images.  This class uses the Augmenter class to generate new
    augmentations on the images produced if wanted.  #TODO replace the augment stuff with the augmenter
    """
    def __init__(self, cvj, image_folder_path=None):
        super().__init__()
        self.__bbox_image_id_2_class_of_object_in_center_dict = {}
        self.__too_big_of_boxes_id = 0
        self.too_big_json = self.create_empty_json()

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

        if image_folder_path != None:
            self.image_folder_path = image_folder_path
        
        

    #TODO: Add MultiProcessing Feature.
    #TODO: reduce the amount of files made.
    def crop_images_bbox_centered(self, save_directory, max_image_size = 768 , min_image_size = 128,
                                  max_bounding_box_size=600, scale = 1, padding = 0,padded_beforehand=False, image_threshold=0, cv2enum=cv2.BORDER_REFLECT, testing=0):
        """
        This method takes every image found in the image_folder_path that was supplied and then gets every
        bounding box that is associated with each image and then crops an image to the size specified to this method,
        with each bounding box being centered in at leas one image.  So this means that if you have 600 bounding boxes
        you will get 600 image crops with each crop having a bounding box in the center.  Now there can be more since this
        also will want to augment images.  Just specify the image threshold and thats how many images each class will have
        where a bounding box of that class is centered inside the JSON.  

        Files produced:
                        1: {TIMESTAMP}_coco_train.json

                            DESCRIPTION : The file that is produced for training.  If the image threshold
                                          was supplied then this file will be stratified and have undersampled
                                          the images more strongly associated with the class that is most dominant
                                          in the json.  It also oversamples the rare classes through augmentation using
                                          the imgaug library
                                        
                                          Format = COCO

                        2: {TIMESTAMP}_stratified_images.json

                            DESCRIPTION : The file that is produced during strafitication.  It has the filepaths
                                          for all the images that are stored in {TIMESTAM}_coco_train.json

                                          Format = None

                                          Keys   = Class Ids

                                          values = all images associated with each class including augments
                                                   up to the threshold

                        3: {TIMESTAMP}_image_class_counts.json

                            DESCRIPTION : This file actually has the filepaths to each image associated with a class.
                                          This is used for counts in the program and is why it is named like that.
                                          
                                          Format = None

                                          keys   = class_id

                                          values = filepaths to all of the images associated with each class including augments
                                                   regardless of the threshold  

                        4: {TIMESTAMP}_augmented_image_class_filepaths.json

                            DESCRIPTION : The file that is produced when the stratification occurs and gives the 
                                          filepaths to the augmented images.  The path's will break if moved so 
                                          it is up to the user how they want to handle this file.

                                          Format = None

                                          Keys   = class ids

                                          values = filepaths to augmented images associated to each class


                        5: {TIMESTAMP}_too_big.json

                            DESCRIPTION : This file contains the images and annotations for any bounding box size that was over the
                                          maximum bounding box size set.
                                          
                                          Format = COCO
        
        Parameters
        ----------

        save_directory : string
             The path to the directory where this will be saved.
            
        max_image_size : int, if a float is supplied then it will be truncated.
             (Default value = 768)
             The maximum size an image can be. If the min_image_size
             is set to the same as this then all images will be square 
             of the size supplied.  Otherwise there will be a range between
             the two.

        min_image_size :  int, if a float is supplied then it will be truncated.
             (Default value = 128)
             The minimum size an image can be. If the min_image_size
             is set to the same as this then all images will be square 
             of the size supplied.  Otherwise there will be a range between
             the two.


        max_bounding_box_size : int, if a float is supplied then it will be truncated.
             (Default value = 600)
             The maximum size a bounding box can be.

        scale : int, if a float is supplied then it will be truncated. 
             (Default value = 1)
             This is used to give take to scale get an image of the
             size of the bounding box scaled and squared.  It is not
             exact scaling but pretty close. The min and max sizes will 
             allow the images to be within range of those, but still
             give some dynamics to the images being cropped if the scale
             is above 1.  However if 1 is used, it is by default, there will
             still be some dynamics in crops unless the min and max image sizes
             are set to the same thing.

        padding : int, if a float is supplied then it will be truncated.
             (Default value = 0)
             This parameter is used to pad images given the opencv padding enum supplied.
             If the padding is zero, it is by defualt, then the cropping will assume that
             your images were padded previously and not try to do it again.  If you receive
             an index out of bounds error, then most likely a bounding box was near and edge
             and there wasn't enough image left to do the crop and increasing this parameter
             should clear that up.

        padded_beforehand : bool
             (Default value = False)
             This tell the method if the image was padded beforehand.  If this is false then the method
             will pad image supplied by the amount of padding that is supplied.  However, the bounding boxes will 
             not be tracked.  The best use for this is just to use the Painter class to pad, track, and inpaint and then
             use the cropper and keep this True and the padding match the padding used in the Painter class.

        image_threshold : int, if a float is supplied then it will be truncated. 
             (Default value = 0)
             This parameter is what makes sure that the amount of images associated
             to each class is the same.  This is what stratifies your images in the JSON
             and if there aren't enough images original to meet the threshold then more
             will be generated using the Augment class.  If the threshold is 0, it is
             by default, then cropping will be unlimited and no stratification will occur.

        cv2enum : int, however this is the same as the opencv border types.
             (Default value = cv2.BORDER_REFLECT)
             This parameter is the same as opencv's border parameters.
             * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
             * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
             * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
             * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
             * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'

        testing : int
             This int is used for unit tests since this method is all about creating crops, writing them out
             and generating files.  We don't need to test over a whole gigantic json all the time as well as images.
             Also we don't want to have to create a new testing json.  So setting this int to anything other than 0 
             makes the method only iterate through that many images.  So if it is set to 2 then it will generate crops
             from 2 images only.

        Returns
        -------
        str :
             Timestamp of when the method was called.
        

        Example
        -------
        
        If I have 7000 bounding boxes 3 classes where class 1 has 4000 bounding boxes, class 2 has 2000, and class 3 has 1000. 
        If the image threshold is set to 3000 then class 1 will not have any augmentation and also only 3000 of the images associated
        with class one will be included in the {TIMESTAMP}_stratified_images_train.json that is produced.  Class 2 will have 1k of augments and
        class 3 will have 2k of augments.  

        These augments rotate through the original images until the limit is reached and uses the 
        Augment class to augment the images.

        NOTE:
        No matter what this method will always generate as many images as there are bounding boxes at a minimum.
        """

        assert (scale >= 1), "The scale must be greater than or equal to one."

        timestamp = str(datetime.datetime.now())[:19]
        timestamp = timestamp.replace(":","_")

        #def __process_chip_image_3(self, output_folder, min_, max_, max_box, scale, padding, cv2enum, image_threshold = 1000, padded_beforehand=False):
        self.__process_chip_image_3(save_directory, min_image_size, max_image_size, max_bounding_box_size, scale, padding, cv2enum, image_threshold, padded_beforehand, testing, timestamp)
        
        return timestamp

    
    def __crop_image_vectorized(self, img, save_directory, bbox_classes, bbox, origin_filename, min_, max_, max_box, scale, padding):


        outfold_box = os.path.join(save_directory,'too_big_of_bounding_boxes')
        subprocess.call(['mkdir', '-p', outfold_box])

        height, width = img.shape[:2]

        # our minimum size we want for detectron is 128x128 because of the the crop that will happen when augmenting and then
        # the downsampling that will occur in detectron
        # max size should be the size of the image.
        min_size = min_
        max_size = max_
        max_bounding_box_size = max_box
        
        bbox = np.asarray(bbox)

        ############################################################### Too Big of Bounding Boxes

        too_big_indices = np.nonzero(np.logical_and(bbox[:, 2] > max_bounding_box_size, bbox[:,3] > max_bounding_box_size))
        too_big_indices = too_big_indices[0]

        good_to_go_indices = np.nonzero(np.logical_and(bbox[:, 2] < max_bounding_box_size, bbox[:,3] < max_bounding_box_size))
        good_to_go_indices = good_to_go_indices[0]

        bboxes_too_big = bbox[too_big_indices]
        bbox_classes_too_big = bbox_classes[too_big_indices]

        good_to_go_bbox = bbox[good_to_go_indices]
        good_to_go_bbox_classes = bbox_classes[good_to_go_indices]


        too_big_diameters = self._get_diameters(bboxes_too_big)
        too_big_patches = too_big_diameters * scale

        good_to_go_diameters = self._get_diameters(good_to_go_bbox)
        good_to_go_patches = good_to_go_diameters * scale

        ###Too_big of bounding boxes being handled from here on until ending comments.
        ## TODO: Locate the bounding boxes on the image.  Right now the bounding box that is saved is 
        ## for the bigger padded image
        ## The x,y,w,h is for readability only.
        x = bboxes_too_big[:, 0] + padding 
        y = bboxes_too_big[:, 1] + padding
        h = bboxes_too_big[:, 3] 
        w = bboxes_too_big[:, 2]


        bboxes_too_big[:, 0] = x 
        bboxes_too_big[:, 1] = y
        bboxes_too_big[:, 3] = h
        bboxes_too_big[:, 2] = w

        midpoint_x = (x+x+w)//2
        midpoint_y = (y+y+h)//2
        
        too_big_patches[too_big_patches > max_size] = max_size

        offset = too_big_patches // 2

        starting_x = (midpoint_x - offset)
        ending_x = (midpoint_x + offset)
        starting_y = (midpoint_y - offset)
        ending_y = (midpoint_y + offset)

        images = {}
        tot_boxes = 0
        idx = 0
        if bboxes_too_big.size != 0:
            for patch in too_big_patches:
                cropped_img = img[int(starting_y[idx]):int(ending_y[idx]), int(starting_x[idx]):int(ending_x[idx])]
                cropped_img = np.asarray(cropped_img)

                imname = "too_big_" + str(idx).zfill(7)+'.png'
                    

                cocoimg = self.entry_img(file_name=imname, height=height,
                                                                width=width, id=self.__too_big_of_boxes_id)


                cocoanns = []
                
                tot_boxes += 1
                cocoanns.append(self.entry_bbox(bbox=bboxes_too_big[idx], class_id=int(bbox_classes_too_big[idx]),
                                                        image_id=self.__too_big_of_boxes_id, id=tot_boxes))

                self.__too_big_of_boxes_id += 1
                cv2.imwrite(os.path.join(outfold_box,imname), cropped_img)

                self.too_big_json['images'].append(cocoimg)
                self.too_big_json['annotations'].extend(cocoanns)
            
            all_category_ids = set([])
            all_category_ids.update([int(vv) for vv in bbox_classes])
            all_category_ids = list(sorted([self.categ_idx_to_coco_categ(vv) for vv in all_category_ids], key=lambda xdic: xdic['id']))
            self.too_big_json['categories'] = all_category_ids
        ##
        ## Too_Big Ending Comment
        ##

        ##
        ### Beginning the usable bounding boxes and crops
        ##

        x = good_to_go_bbox[:, 0] + padding 
        y = good_to_go_bbox[:, 1] + padding
        h = good_to_go_bbox[:, 3] 
        w = good_to_go_bbox[:, 2]

        good_to_go_bbox[:, 0] = x 
        good_to_go_bbox[:, 1] = y
        good_to_go_bbox[:, 3] = h
        good_to_go_bbox[:, 2] = w

        midpoint_x = (x+x+w)//2
        midpoint_y = (y+y+h)//2
        
        good_to_go_patches[good_to_go_patches < min_size] = min_size
        good_to_go_patches[good_to_go_patches > max_size] = max_size

        offset = good_to_go_patches // 2

        starting_x = (midpoint_x - offset)
        ending_x = (midpoint_x + offset)
        starting_y = (midpoint_y - offset)
        ending_y = (midpoint_y + offset)

        images = {}
        idx = 0
        for patch in good_to_go_patches:
            cropped_img = img[int(starting_y[idx]):int(ending_y[idx]), int(starting_x[idx]):int(ending_x[idx])]
            cropped_img = np.asarray(cropped_img)
            
            cropped_height, cropped_width = cropped_img.shape[:2]
            if cropped_height < min_size or cropped_width < min_size:
                height_difference = abs(cropped_height - min_size)
                width_difference = abs(cropped_width - min_size)

                ending_y[idx] += height_difference
                ending_x[idx] += width_difference 

                cropped_img = img[int(starting_y[idx]):int(ending_y[idx]), int(starting_x[idx]):int(ending_x[idx])]
                cropped_img = np.asarray(cropped_img)
            
            images[ idx ] = cropped_img

            idx +=1

        ##
        ### Ending Usable bounding boxes and crops
        ##

        return images, good_to_go_patches, good_to_go_bbox, starting_x, ending_x, starting_y, ending_y, good_to_go_bbox_classes


    ### Modified code from wv_utils
    #def chip_image_2(img, bboxcoords, bboxclasses, patchshape, strideshape):
    ###This will need to be made for multiprocessing.
    def __chip_image_3(self, img_id, img, img_id_2_anns_dict, save_directory,origin_filename, min_, max_, max_box, scale=5, padding=3000):

        try:
            bboxcoords = [ann["bbox"] for ann in img_id_2_anns_dict[img_id]]
            bboxclasses = [ann["category_id"] for ann in img_id_2_anns_dict[img_id]]

        except KeyError:
            return None, None, None, None

        bboxcoords = np.asarray(bboxcoords)
        bboxclasses = np.asarray(bboxclasses)
        bbox_image_id_2_class_of_object_in_center_dict = {}
        total_boxes = {}
        total_classes = {}

        ACCEPTABLE_VISIBLE_FRACTION = 0.75

        ############### First lets crop the images
        images, patches, bboxcoords, starting_x, ending_x, starting_y, ending_y, bboxclasses = self.__crop_image_vectorized(img,save_directory,
                                                                                                                            bboxclasses,bboxcoords,origin_filename,
                                                                                                                            min_, max_, max_box,
                                                                                                                            scale=scale, padding=padding)
        #changing the x,y,w,h format to x0,y0,x1,y1 format # The x and y are padded also.
        bboxcoords[:,2] = bboxcoords[:, 0] + bboxcoords[:, 2] #all of the x2's
        bboxcoords[:,3] = bboxcoords[:, 1] + bboxcoords[:, 3] #all of the y2's

        ############ all operations have been done with performance in mind.  however I can't slice the image this way and must use the for loop
        patch_idx = -1
        for patch_size in tqdm(patches):
            patch_idx += 1
            c0_ = starting_x[patch_idx] # these where the indices for the crops
            r0_ = starting_y[patch_idx] #
            c1_ = ending_x[patch_idx]   #
            r1_ = ending_y[patch_idx]   #

            newbbox = np.ones_like(bboxcoords) * -1

            #if patch_idx not in bbox_image_id_2_class_of_object_in_center_dict:
            bbox_image_id_2_class_of_object_in_center_dict[patch_idx] = bboxclasses[patch_idx]
            '''
            np.logical_and(bboxcoords[:,0]<c1_,bboxcoords[:,0]>c0_)
            This is saying that if the bottom right corner x of the bbox is less than the bottom right corner x of the image
            and the bboxs top left corner x is greater than the top left corner x of the image, that it returns true.

            In other words this is true if the bounding box x corner is not outside of the image return true.  The bounding box top left x
            would be the point where it begins to draw.  So if this point is greater than the c1_, which is the furthest x coordinate of the new image,
            then the bounding box wouldn't be in this image since it starts in another.  Same goes for the left side. 
            If the bounding boxes x is less than the c0_, which is the left most x coordinate of the new image, then the bounding box is not starting in the image.

            So that checks that it is within the image by reversing what the logic I said previously.  In other words, instead of checking to see if it is outside
            the image the np.logical_and(bboxcoords[:,0]<c1_,bboxcoords[:,0]>c0_) checks that it is INSIDE the image.

            np.logical_and(bboxcoords[:,2]<c1_,bboxcoords[:,2]>c0_) does the same thing except it is checking to make sure the bounding box ends INSIDE the image.

            The logical or is saying that if either returns true then that will also return true.  This is good because based on the two logical ands we know
            that the bounding box may be cut off because of where it starts.  Lets say for simplicity that the 2nd and statement is true and the first is false.

            This means that the bounding box starts outside of the image to the left, but ends inside the image.  The bbox cant start on the rightmost and then
            still end within the image. So the statement before is the only thing that can happen in that scenario.

            Now if the 2nd is false and the first is true then the bouning box starts in the image and then extends off of it.

            Since this is done as a matrix operation then taking np.nonzero() will return the cols of where the statement was true at x_[1] and the row indices
            where they were true at x_[0].  
            '''
            x_ = np.nonzero(np.logical_or( np.logical_and(bboxcoords[:,0]<c1_,bboxcoords[:,0]>c0_),
                                            np.logical_and(bboxcoords[:,2]<c1_,bboxcoords[:,2]>c0_) ))

            
            x_ = x_[0] # just the row indices (along the first axis)

            '''
            This is going to get all of the bboxs that are in the image or at least partially in the image. Since there is probably just one row for each
            bounding box this will just get all of the bounding boxes as described in the previous sentence
            '''
            out = bboxcoords[x_]


            '''
            Same logic is applied like before except now there is checking if the y values are in the image partially or fully.
            '''
            y_ = np.nonzero(np.logical_or( np.logical_and(out[:,1]<r1_,out[:,1]>r0_),
                                            np.logical_and(out[:,3]<r1_,out[:,3]>r0_) ))


            assert len(y_) == 1, str(len(y_))
            y_ = y_[0] # just the row indices (along the first axis)

            '''
            This will further reduce the amount of bounding boxes that met the requirements of the x coordinates.  This is becuase some will fail to have the y values
            in the image partially or fully.
            '''
            outn = out[y_]
            x_ = x_[y_] # subselect the row indices


            '''
            This is saying that if the bounding box started to the left outside the image to remake it to starting at the edge of the image which is 0 and can
            only go the the maximum of the image.  That is just the first clip, but they all essentially say the same thing.  If it has some portion outside the image
            clip and close the bounding box to be inside the image.
            '''
            # this is where the bounding boxes are translated!!
            # since we shortened the image by c0_ to crop the image that means we need to subtract it from the bounding box since the bounding box
            # still has coordinates for the original image yet the image is smaller.  Which is smaller by the amount of c0_, r0_
            
            new_bbox_x = outn[:,0] - c0_
            new_bbox_y = outn[:,1] - r0_
            new_bbox_x1 = outn[:,2] - c0_
            new_bbox_y1 = outn[:,3] - r0_

            out = np.transpose(np.vstack((np.clip(new_bbox_x,0,patch_size),
                                            np.clip(new_bbox_y,0,patch_size),
                                            np.clip(new_bbox_x1,0,patch_size),
                                            np.clip(new_bbox_y1,0,patch_size)))) #c0_ is the x of the image and r0_ is the y of the image

            '''
            Now just copy the clipped bounding boxes in the newbbox matrix at the row indices that survived the logical operations.
            '''
            newbbox[x_] = np.copy(out)

            assert len(newbbox.shape) == 2 and newbbox.shape[1] == 4, str(newbbox.shape)
            maxnbox = np.amax(newbbox, axis=1) #No idea why this but it is getting the maximum value of each row.
            assert np.count_nonzero(maxnbox > -0.95) == out.shape[0], \
                str(np.count_nonzero(maxnbox > -0.95))+', '+str(out.shape)

            for kk_ in range(bboxcoords.shape[0]):
                assert bboxcoords[kk_,3] >= bboxcoords[kk_,1], str(bboxcoords[kk_])
                assert bboxcoords[kk_,2] >= bboxcoords[kk_,0], str(bboxcoords[kk_])

            '''
            since the format the bboxcoords are in is (x0,y0) = top left corner = bboxcoords[0], bboxcoords[1] and (x1,y1) =  bottom right corner
            = bboxcoords[2], bboxcoords[3].  They are converted to the width and height of their original coords and then the next line gets the new
            width and height of the newly clipped bbox coordinates.

            however since there is multiplication this is definitly comparing the areas.  so oldwh and newwh are actually areas, not width and height.
            '''
            oldwh = (bboxcoords[:,2]-bboxcoords[:,0]) * (bboxcoords[:,3]-bboxcoords[:,1])
            newwh = (   newbbox[:,2]-   newbbox[:,0]) * (   newbbox[:,3]-   newbbox[:,1])


            newfrac = np.divide(newwh, np.maximum(oldwh, np.ones_like(oldwh)*1e-9)) #getting the percentage of area that is still within the image
                                                                                    #by dividing the original area by the new area which is on the image.
            
            '''
            This gets all of the indices of the bounding boxes that the visible area on the image within a set threshold and had an area that was greater than 10
            '''
            chosenid = np.nonzero(np.logical_and(newfrac > ACCEPTABLE_VISIBLE_FRACTION, oldwh > 10.))[0]


            out         =     newbbox[chosenid]

            box_classes = bboxclasses[chosenid] # this will be the bbox classes for each of those bboxs.  This is necessary for iding the object.
            if len(out) > 0:
                assert np.amin(out) > -0.01, str(np.amin(out))

            numafteraspectfiltering = int(len(chosenid)) #just how many bounding boxes survived

            '''
            This is tallying up how many boxes are within the image/patch and how many classes are in the image/patch it's also shoving
            the found bounding boxes in something that will be passed.
            '''
            if numafteraspectfiltering > 0:
                total_boxes[patch_idx]   = out.copy()
                total_classes[patch_idx] = np.int64(box_classes)
            else:
                total_boxes[patch_idx] = np.array([[0,0,0,0]])
                total_classes[patch_idx] = np.array([0])

            # Uncomment this block to see if it is not working correctly.  Most likely if it is not working correctly
            # then your problem lies within __crop_image_vectorized() and your image is not a padded one.  If that is the case
            # then the padding variable must be 0
            '''
            ipatch = images[patch_idx]
            for bbox in out:
                cv2.rectangle(ipatch, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color= (255,0,0), thickness=1)

            cv2.imshow("test", ipatch)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            

        return images, total_boxes, total_classes, bbox_image_id_2_class_of_object_in_center_dict



    def __process_chip_image_3(self, output_folder, min_, max_, max_box, scale, padding, cv2enum, image_threshold, padded_beforehand, testing, timestamp):

        assert os.path.isdir(self.image_folder_path), self.image_folder_path

        outfold_trn = os.path.join(output_folder, str(timestamp) +'_trn')
        too_big_save = os.path.join(output_folder, str(timestamp) +'_too_big.json')
        image_class_path = os.path.join(output_folder, str(timestamp) + "_image_class_counts.json")
        augmented_image_class_path = os.path.join(output_folder, str(timestamp) + "_augmented_image_class_filepaths.json")
        stratified_images_path = os.path.join(output_folder, str(timestamp) + "_stratified_images.json")
        subprocess.call(['mkdir', '-p', outfold_trn])

        img_id_2_anns_dict = self.get_image_id_2_anns()
        filename2imageid = self.get_filename_2_image_id()
        image_id_filepath_dict = self.get_image_id_2_filepath()

        train_json = self.create_empty_json()

        class_id_2_filepath = {}
        filenames_to_origin_image_ids = {}

        all_category_ids = set([])
        tot_chips = 0
        tot_box = 0
        ids_in_filepath = list(image_id_filepath_dict.keys())
        
        #################################################################################################################################
        #
        #                                                   CROPPING
        #
        #################################################################################################################################

        count = 0
        for id in tqdm(ids_in_filepath): #tqdm
            #print("The image_id being processed right now is {}\n".format(id))

            ################################################# TESTING CODE

            if testing > 0:
                if count > testing - 1: # this is minused since testing of 2 means test on 2 images.  
                    break
                count += 1

            ################################################# TESTING CODE

            img = cv2.imread(image_id_filepath_dict[id], cv2.IMREAD_COLOR)

            if padded_beforehand == False:
                img = cv2.copyMakeBorder(img,padding,padding,padding,padding, cv2enum)
            try:
                height, width = img.shape[:2]
            except:
                filename = os.path.basename(image_id_filepath_dict[id])
                print("There was an error reading file {} logging it to the file \"error_log.log now\"".format(filename))
                if os.path.isfile("error_log.log"):
                    with open("error_log.log", 'a') as file:
                        file.write("\nRead error occurred in __crop_image_vectorized.  Failure to read shape of {}".format(filename))  
                else:
                    with open("error_log.log", 'w') as file:
                        file.write("\nRead error occurred in __crop_image_vectorized.  Failure to read shape of {}".format(filename))  
                continue

            filename = os.path.basename(image_id_filepath_dict[id])

            '''
            img_center_objects are the object classes that are in the center of each image.  The local image id's are the keys to 
            the classes. so it's just 1,2,3,4... Which this information will be lost unless handled now.
            '''
            im,box,classes_final, img_center_objects = self.__chip_image_3(id, img, img_id_2_anns_dict, output_folder, filename, min_, max_, max_box, scale, padding)

            ###########################################################################
            # This below is because chip_image_3 will return None values on a key error
            # which is caused when filepath has some images which the json file has 
            # no annotations for.  This will definitely happen when trying to 
            # single out a class.
            ##########################################################################
            if im == None or box == None or classes_final == None or img_center_objects == None:
                print("KeyError exception.  No img_id {} found in json.\n".format(id))
                continue

            for idx in im:
                image = im[idx]
                height, width = image.shape[:2]
                cocoboxes = []
                cococlass = []
                for bbidx in range(len(box[idx])):
                    bbox = box[idx][bbidx]
                    assert len(bbox) == 4, str(bbox)
                    # these bboxes are in format (col0, row0, col1, row1)
                    assert bbox[2] >= bbox[0] and bbox[3] >= bbox[1], str(bbox)
                    # convert to coco format (col0, row0, width, height)
                    cwidth  = float(bbox[2]-bbox[0])
                    cheight = float(bbox[3]-bbox[1])
                    if cwidth > 1. and cheight > 1.:
                        cocoboxes.append((bbox[0], bbox[1], cwidth, cheight))
                        cococlass.append(int(classes_final[idx][bbidx]))

                    ################################################# TESTING CODE

                    if testing > 0:

                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + cwidth),int(cheight + bbox[1])), color= (255,0,0), thickness=1)

                    ################################################# TESTING CODE
                '''
                # DEBUGGING CODE
                print("amount of boxes for image being looked at is {}".format(len(cocoboxes)))
                print("coco class")
                print(cococlass)
                input()
                '''
                
                all_category_ids.update([int(vv) for vv in cococlass])

                '''
                # DEBUGGING CODE
                print("class id for image being looked at is")
                print(str(img_center_objects[idx]))
                '''

                if len(cocoboxes) >= 1: # if there is at least one valid bounding box
                    tot_chips += 1

                    imname = str(os.path.basename(image_id_filepath_dict[id])).split('.')[0] + '_' + str(tot_chips).zfill(7)+'.png'

                    if str(img_center_objects[idx]) not in class_id_2_filepath:
                        class_id_2_filepath[str(img_center_objects[idx])] = [os.path.join(outfold_trn,imname),]
                        filenames_to_origin_image_ids[imname] = id
                    else:
                        class_id_2_filepath[str(img_center_objects[idx])].append(os.path.join(outfold_trn,imname))
                        filenames_to_origin_image_ids[imname] = id  

                    cocoimg = self.entry_img(file_name=imname, height=height, width=width, id=tot_chips)
                    cocoanns = []
                    for ii in range(len(cocoboxes)): 
                        tot_box += 1
                        cocoanns.append(self.entry_bbox(bbox=cocoboxes[ii], class_id=int(cococlass[ii]),
                                                            image_id=tot_chips, id=tot_box))

                    cv2.imwrite(os.path.join(outfold_trn,imname), image)

                    train_json['images'].append(cocoimg)
                    train_json['annotations'].extend(cocoanns)
        '''
        # DEBUGGING CODE
        print("all_category_ids")
        print(all_category_ids)
        '''
        assert len(all_category_ids) > 0, str(all_category_ids)
        all_category_ids = list(sorted([self.categ_idx_to_coco_categ(vv) for vv in all_category_ids], key=lambda xdic: xdic['id']))
        train_json['categories'] = all_category_ids

        '''
        # DEBUGGING CODE
        print("inside cropper")
        bbox_classeid = self.get_bbox_count_by_class_dict(json_data=train_json) 
        print(bbox_classeid.keys())
        input()
        print("the id's found before are")
        print(class_id_2_filepath.keys())
        print("the counts of images for those ids are")
        self.get_image_counts_by_class_id(class_id_2_filepath)
        input()
        '''

        outjson_tr = os.path.join(output_folder,str(timestamp) + '_coco_train.json')


        #################################################################################################################################
        #
        #                                                   AUGMENTING
        #
        #################################################################################################################################
        class_id_2_augmented_images = {}
        stratified_class_id_2_filepath = {}
        if image_threshold > 0:
            '''
            I'm rewriting a few dictionary variables since the initial tracking of the annotations is done and I need to augment and retrack.
            '''
            image_id_2_anns_dict_2 = self.get_image_id_2_anns(json_data=train_json)
            filename2imageid = self.get_filename_2_image_id(json_data=train_json)


            '''
            # DEBUGGING CODE
            catlist =[]
            for cat in train_json["categories"]:
                catlist.append(int(cat["id"]))
            print(sorted(catlist))
            input()
            '''
            
            catlist = [cat["id"] for cat in train_json["categories"]]

            # This is actually just the count of how many files are associated with the class id.  The function definitions are not very readable to 
            # understand this at this point in time. 6/10/2018
            cat_ids_needing_augments = self.__get_cat_ids_needing_augments(class_id_2_filepath, image_threshold, catlist)

            for id in tqdm(cat_ids_needing_augments):
                print("\ncategory id being augmented is {}\n".format(id)) 
                #print(class_id_2_filepath.keys())
                
                amount_of_images = len(cat_ids_needing_augments[id])
                img_file_paths = class_id_2_filepath[str(id)]
                filenames = [os.path.basename(filepath) for filepath in img_file_paths]
                #print(filenames)
                i = 0
                iteration_count = 0
                while amount_of_images < image_threshold:
                    
                    img_id = filename2imageid[filenames[i]]
                    origin_img_id = filenames_to_origin_image_ids[filenames[i]]
                    patch = cv2.imread(img_file_paths[i], cv2.IMREAD_COLOR)
                    
                    #We already know that these bboxes have a width and height greater than 0 so, no need to check.
                    bboxes = [ann["bbox"] for ann in image_id_2_anns_dict_2[img_id]]
                    bbox_classes = [ann["category_id"] for ann in image_id_2_anns_dict_2[img_id]]
                    bboxes = np.asarray(bboxes)

                    ########################################################################################## Augmentation
                    self = Augmenter(self)
                    patch = cv2.imread(img_file_paths[i], cv2.IMREAD_COLOR)
                    patch, bboxes = self.transf_one_patch(patch,bboxes)
                    #transf_one_patch(patch, bboxes)
                    ########################################################################################## Augmentation

                    continue_flag = False
                    for bbox in bboxes:
                        if bbox[2] <= 0 or bbox[3] <= 0:
                            print("Some width or height were less than or equal to zero.")
                            continue_flag = True
                            break
                    
                    if continue_flag:
                        i += 1
                        if i >= len(cat_ids_needing_augments[id]):
                            i = 0
                        continue

                    
                    # This code just helps debug by visualizing the image.  This was mainly used for the dataset_medifor_patches.py script
                    '''
                    if cat_ids_needing_augments[id] > 1:
                        self.visualize_bboxes_with_image_and_bboxes(patch, bboxes)
                    '''

                    tot_chips += 1

                    imname = "augment_" + str(os.path.basename(image_id_filepath_dict[origin_img_id])).split('.')[0] + '_' + str(tot_chips).zfill(7)+'.png'
                    class_id_2_filepath[str(id)].append(os.path.join(outfold_trn,imname))

                    cocoimg = self.entry_img(file_name=imname, height=patch.shape[0],
                                                            width=patch.shape[1], id=tot_chips)
                    cocoanns = []
                    for ii in range(len(bboxes)):
                        tot_box += 1
                        cocoanns.append(self.entry_bbox(bbox=bboxes[ii], class_id=int(bbox_classes[ii]),
                                                            image_id=tot_chips, id=tot_box))

                    if str(id) not in class_id_2_augmented_images:
                        class_id_2_augmented_images[str(id)] = [os.path.join(outfold_trn,imname),]
                    else:
                        class_id_2_augmented_images[str(id)].append(os.path.join(outfold_trn,imname))

                    cv2.imwrite(os.path.join(outfold_trn,imname), patch)

                    train_json['images'].append(cocoimg)
                    train_json['annotations'].extend(cocoanns)

                    i += 1 
                    
                    # this is needed so that the images will rotate through the augmentation process. If the iteration is greater than the amount
                    # of images avaible.  Go to the beginning.
                    if i >= len(cat_ids_needing_augments[id]):
                        i = 0

                    amount_of_images += 1
                    iteration_count += 1

                    if iteration_count % 1000 == 0:
                        print("augmenting is currently at iteration {}".format(iteration_count))
                        if iteration_count == image_threshold:
                            print("Completed augmenting {} images for category id {}".format(iteration_count, id))
            '''
            # DEBUGGING CODE
            print("before stratification")
            bbox_classeid = self.get_bbox_count_by_class_dict(json_data=train_json)
            print(bbox_classeid.keys())
            input()
            '''
            '''
            # DEBUGGING CODE
            catlist =[]
            for id, file_list in class_id_2_filepath.items():
                catlist.append(int(id))
            
            print(sorted(catlist))
            input()
            '''
            '''
            # DEBUGGING CODE
            catlist =[]
            for cat in train_json["categories"]:
                catlist.append(int(cat["id"]))
            
            print(sorted(catlist))
            input()
            '''

            self = Cropper(self)
            train_json, stratified_class_id_2_filepath = self.__stratify_json(train_json, class_id_2_filepath, image_threshold) 
            print("JSON file image stratification complete")
        '''
        # DEBUGGING CODE
        print("saving json")
        print("before save")
        bbox_classeid = self.get_bbox_count_by_class_dict(json_data=train_json)
        print(bbox_classeid.keys())
        input()
        '''

        print("Saving file {} at filepath {}".format(os.path.basename(image_class_path), image_class_path))
        with open(image_class_path, 'w') as outfile:
            json.dump(class_id_2_filepath, outfile)

        print("Saving file {} at filepath {}".format(os.path.basename(outjson_tr), outjson_tr))
        with open(outjson_tr, 'w') as outfile:
            json.dump(train_json, outfile)

        print("Saving file {} at filepath {}".format(os.path.basename(too_big_save), too_big_save))
        with open(too_big_save, 'w') as outfile:
            json.dump(self.too_big_json, outfile)

        print("Saving file {} at filepath {}".format(os.path.basename(augmented_image_class_path), augmented_image_class_path))
        with open(augmented_image_class_path, 'w') as outfile:
            json.dump(class_id_2_augmented_images, outfile)

        print("Saving file {} at filepath {}".format(os.path.basename(stratified_images_path), stratified_images_path))
        with open(stratified_images_path, 'w') as outfile:
            json.dump(stratified_class_id_2_filepath, outfile)

        assert os.path.isfile(outjson_tr), outjson_tr

        return train_json, timestamp

    def __get_cat_ids_needing_augments(self, class_file_path, image_threshold, cats_in_json ):
        """
        This function just counts the amount of bounding boxes and if they are less than the
        threshold given), then they delete the class id associated with that class.

        """
        if class_file_path != None:
            cat_id_needing_augments = copy.deepcopy(class_file_path)

            ### clearing up any additional classes that don't have annotations
            for id in class_file_path:
                if int(id) not in cats_in_json:
                    del cat_id_needing_augments[id]
            ###
            

            for id in cats_in_json: # class_file_path:
                if len(class_file_path[str(id)]) > image_threshold:
                    try:
                        del cat_id_needing_augments[id]
                        print("class id {} does not need augmentation.".format(id))
                    except KeyError:
                        print("Could not find key {} when deleting".format(id))

        return cat_id_needing_augments
            
    def __stratify_json(self, json_data, class_id_2_filepath, threshold):


        cat_2_anns_dict =       self.get_class_id_2_anns_dict(json_data=json_data)
        image_id_2_anns =       self.get_image_id_2_anns(json_data=json_data)
        filename_2_image_id =   self.get_filename_2_image_id(json_data=json_data)
        img_attribs_dict =      self.get_image_id_2_image_attribs(json_data=json_data)

        new_json = self.create_empty_json()
        
        categories = [self.categ_idx_to_coco_categ(cat) for cat in list(cat_2_anns_dict.keys())]
        new_json["categories"] = categories

        stratified_class_id_2_filepath = {}
        for cat in new_json["categories"]:
            id = cat["id"]
            class_id_images_list = class_id_2_filepath[str(id)]
            iteration_count = 0
            stratified_images = []
            #print("the amount of images for class id {} is {}".format(id, len(class_id_images_list)))
            while iteration_count < threshold:
                stratified_images.append(class_id_images_list[iteration_count])
                iteration_count += 1
            
            stratified_class_id_2_filepath[str(id)] = stratified_images

            '''
            # DEBUGGING CODE
            print("inside strafity after appending images")
            print(stratified_class_id_2_filepath.keys())
            input()
            '''

            for filepath in stratified_images:
                name = os.path.basename(filepath)
                img_id = filename_2_image_id[name]
                new_json['images'].append(img_attribs_dict[img_id])
                
                for ann in image_id_2_anns[img_id]:
                    new_json["annotations"].append(ann)

        
        #cat_2_anns_dict = get_category_id2anns_dict(new_json)
        #print(cat_2_anns_dict.keys())
        #input()
        #print("end of strat")
        return new_json, stratified_class_id_2_filepath
        


if __name__ == "__main__":
    print("Do not call this script as a main function.  This is a script to house functions driven by another script.")
