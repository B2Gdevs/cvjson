
# import sys
# import os
# sys.path.insert(0, os.path.abspath('../cvjson/cvjson/'))
# sys.path.insert(0, os.path.abspath('../extensions/'))
# print(sys.path)
import os
import cv2
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from ..cvj import CVJ      
from itertools import repeat
import subprocess
import shutil
import pickle
# from enum import Enum #TODO: Make good enums
 

class Painter(CVJ):
    """
    The Painter class is used to do painting on
    the images such as inpainting and hopefully
    outpainting.  This class relies on GMIC

    Enums
    -----
    Generation method enums:
                            * INPAINT, using the multiscale patch based inpainted method when 
                              generating data.

    Sample type enums:
                     * POSITIVE, deals with positives samples
                     * NEGATIVE, deals with negatives samples
                     * ALL, deals with both sample types

    """
    # TODO: make official enums
    # Generation method Enums 10-30
    INPAINT = 0

    # Example types 0 or 1
    POSTIVE = 1
    NEGATIVE = 0

    # Use for both POSITIVE and NEGATIVE samples, 9000 is the number paying homage to Dragonballz
    ALL = 9000

    def __init__(self, cvj, image_folder_path=None): 
        super().__init__()

        if image_folder_path != None:
            self.image_folder_path = image_folder_path # Calling the setter
        
        self.negative_filepaths = []
        self.positive_filepaths = []
        #self.temp_filepaths = []

        # This is how many iterations until the checkpoints update save.
        self.checkpoint_iter_thresh = 0
        self.checkpoint_iter = 0

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

    # def padded_inpainting(self, save_directory, n_cores, padding=3000, inplant_orignal=True): 
    #     """
    #     This function drives the "_reflected_inpaint()" with multiple cores and runs GMIC on each one.
    #     This function generates padded images with reflection and multiscale patchbased inpainting.
    #     This actually inpaints every bounding box including the original ones in a padded image.  It then
    #     takes the region where all of the orignal (not reflected) boxes were and overwrites it with the original
    #     image if inplant_original is True. Which by default it is.
        
    #     This is supposed to be used in conjunction with chip_image_3() to develop crops from these padded
    #     images, and then further augmentation can be done with transf_one_patch() 


    #     Parameters
    #     ----------
    #     save_directory : string
    #          The path to the source directory where all your files will be save.  folders will be created accordingly
            
    #     n_cores : int
    #          The number of cpus wanted to be used
            
    #     padding : int
    #          (Default value = 3000)
    #          How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

    #     inplant_orignal : bool
    #          (Default value = True)
    #          A bool that, if true, puts the original image back in the center of the newly inpainted image.

    #     Returns
    #     -------

    #     """

    #     # TODO: add transf_one_patch from dataset_medifor_patches.py


    #     filename2imageid = self.get_filename_2_image_id()
    #     img_id_2_anns_dict = self.get_image_id_2_anns()

    #     if  not os.path.isdir(save_directory):
    #         os.makedirs(save_directory)
    #     else:
            
    #         file_names_in_finished_folder = os.listdir(save_directory)
    #         print("The amount of items in the final saving directory is {}".format(len(file_names_in_finished_folder)))

    #         image_ids_in_finished_folder = []

    #         for name in file_names_in_finished_folder:
    #             try:
    #                 image_ids_in_finished_folder.append(filename2imageid[name])
    #             except KeyError:
    #                 print("couldn't find image {}".format(name))   
            

    #         for img_id in image_ids_in_finished_folder:
    #             try:
    #                 del img_id_2_anns_dict[img_id]
    #                 print("deleted image id {}".format(img_id))
    #             except KeyError:
    #                 print("Tried to delete image id: {} because it found an image associated with that id in the completed pictures directory".format(img_id))
        
    #     print("The amount of images left to inpaint are {}".format(len(img_id_2_anns_dict.keys())))
    #     pool = Pool(n_cores)
    #     pool.starmap(self._reflected_inpaint, zip(img_id_2_anns_dict, repeat(save_directory), repeat(padding), repeat(inplant_orignal)))


    def _reflected_inpaint(self, img_id, reflected_imgs_save_path, padding, inplant_orignal, sample_type, negative_paths, positive_paths):
        """

        This is paired with it's multiprocessing driver function. "multiprocessing_padded_inpainting()

        Parameters
        ----------
        img_id : int
             This parameter is the image_id of the image that is
             to be inpainted using multiscale patch-based inpainting.
            
        reflected_imgs_save_path : string
             The path to the source directory where all your files will be save.  Folders will be created accordingly

        padding : int
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        inplant_orignal : bool
             A bool that, if true, puts the original image back in the center of the newly inpainted image.

        sample_type : int
             This variable is an int/enum/class variable that is used to tell this method what type of generations should occur
             it it is Painter.NEGATIVE then it will paint over all of the objects in the image.  If it is Painter.POSITIVE then
             the inpainting method will leave the objects in the image

        negative_paths : listproxy
             This data type is created from Multiprocessing.Manager().list().  
             This is actually a list proxy and is used to share the negative sample file paths between processes.
             This is saved in checkpoints also.

        positive_paths : listproxy
             This data type is created from Multiprocessing.Manager().list().  
             This is actually a list proxy and is used to share the positive sample file paths between processes.
             This is saved in checkpoints also.
            

        Returns
        -------
        listproxy : listproxy
             The first return value is a list proxy named "negative_filepaths" created from Multiprocessing.Manager().list().  
             This is actually a list proxy and is used to share the negative sample file paths between processes.
             This is saved in checkpoints also.

        listproxy : positive_filepaths
             The second return value is a list proxy named "positive_filepaths" created from Multiprocessing.Manager().list().  
             This is actually a list proxy and is used to share the negative sample file paths between processes.
             This is saved in checkpoints also.

        """
        self.negative_filepaths = negative_paths
        self.positive_filepaths = positive_paths

        # Create temp path to hold all images!!!
        # Do not make images go directly in to the image folder path without a directory
        # this will cause the users images to become tainted with negative or positive examples
        temp_working_path = os.path.join(reflected_imgs_save_path, "temp")


        if not os.path.isdir(temp_working_path):
            os.makedirs(temp_working_path)

        image_id_to_filepath_dict = self.get_image_id_2_filepath()

        path_to_drawn_img, mask_color, img = self._draw_on_on_big_image(img_id,  temp_working_path, padding)

        if sample_type == Painter.POSTIVE:
            file_name = image_id_to_filepath_dict[img_id]
            file_name = "positive_" + os.path.basename(file_name)
            #file_path_to_save = os.path.join(temp_working_path, "inpainted_images")
            file_path_and_name = os.path.join(temp_working_path, file_name)

            self.positive_filepaths.append(file_path_and_name)

        if sample_type == Painter.NEGATIVE:
            file_name = image_id_to_filepath_dict[img_id]
            file_name = "negative_" + os.path.basename(file_name)
            #file_path_to_save = os.path.join(temp_working_path, "inpainted_images")
            file_path_and_name = os.path.join(temp_working_path, file_name)

            self.negative_filepaths.append(file_path_and_name)
        
        if self.checkpoint_iter >= self.checkpoint_iter_thresh:
            self.update_checkpoint()
            self.checkpoint_iter = 0

        if not os.path.isdir(temp_working_path):
            os.makedirs(temp_working_path)

        
        print("right before gmic path to drawn image = {}".format(path_to_drawn_img))

        # FILES THAT END IN TIF OR TIFF WILL BE SHOWING UP AS NOTHING IF YOU HAVE GMIC OUTPUT THAT.  IT'S AN UNRECOGNIZED FILE TYPE.
        subprocess.call("gmic " + '\"' + path_to_drawn_img + '\"' + " --select_color 0,255,0,0 --inpaint_patchmatch[0] [1]  -o[-1] " + '\"' + file_path_and_name + '\"' , shell=True)

        if inplant_orignal:
            #file_path_to_save = os.path.join(temp_working_path, "inpainted_with_regular_image")
            #file_name_and_path_ = os.path.join(file_path_to_save, file_name)

            #if not os.path.isdir(file_path_to_save):
            #    os.makedirs(file_path_to_save)

            image_height, image_width = img.shape[:2]

            reflected_img_with_inpainting = cv2.imread(file_path_and_name, cv2.IMREAD_COLOR)

            #This line of code is how to replace the middle of the image back with the original image
            reflected_img_with_inpainting[padding:int(image_height + padding), padding:int(image_width + padding)] = img

            cv2.imwrite(file_path_and_name,reflected_img_with_inpainting)

        return self.negative_filepaths, self.positive_filepaths



    def _draw_on_on_big_image(self, img_id, reflected_imgs_save_path, padding):
        """
        This method draws the red masks that gmic uses to produce the inpainting results. Based on the padding given
        if there is any.  

        Parameters
        ----------     
        img_id : int
             This parameter is the image_id of the image that is
             to be inpainted using multiscale patch-based inpainting.
            
        reflected_imgs_save_path : string
             The path to the source directory where all your files will be save.  Folders will be created accordingly.

        padding : int
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        Returns
        -------
        string : string
             The first return value is named "save_path_for_image" and this save path
             is where the files will be saved.
             
        tuple : tuple
             This is second return value which is a color tuple named "mask_color" in RGB format that is used for the 
             masking with inpainting, but is also the color that will be drawn on the image.

        numpy array : numpy array
             This is the third return value named "img".  This is the numpy array for the image which what supplied.  
             The reflected, drawn image is not returned.  

        """

        image_id_to_filepath = self.get_image_id_2_filepath()
        img_id_2_anns_dict = self.get_image_id_2_anns()
        
        bboxcoords = np.asarray([ann["bbox"] for ann in img_id_2_anns_dict[img_id]]) #gathering bboxes

       
        x, y, w , h,  img = self._get_padded_indices_of_bbox(image_id_to_filepath[img_id], bboxcoords, padding)
        R_right, R_left, R_top, R_bottom , R_top_left, R_top_right, R_bottom_left, R_bottom_right = self._get_reflected_bounding_boxes(bboxcoords, img, padding=padding)
        
        mask_color = (0,0,255)

        ######### DRAWING RED MASKS ON BBOXES FOR REFLECTED IMAGE ###########
        img[img>254]=254

        origin_height, origin_width  = img.shape[:2]

        if padding > 0:
            reflected_img = cv2.copyMakeBorder(img,padding,padding,padding,padding, cv2.BORDER_REFLECT)
            
            x = R_right[:, 0] 
            y = R_right[:, 1] 
            h = R_right[:, 3] 
            w = R_right[:, 2] 
            
            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

            x = R_left[:, 0] 
            y = R_left[:, 1] 
            h = R_left[:, 3] 
            w = R_left[:, 2] 

            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

            x = R_top[:, 0] 
            y = R_top[:, 1] 
            h = R_top[:, 3] 
            w = R_top[:, 2] 

            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

            x = R_bottom[:, 0] 
            y = R_bottom[:, 1] 
            h = R_bottom[:, 3] 
            w = R_bottom[:, 2] 
            
            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

            x = R_top_left[:, 0] 
            y = R_top_left[:, 1] 
            h = R_top_left[:, 3] 
            w = R_top_left[:, 2] 
            
            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

            x = R_top_right[:, 0] 
            y = R_top_right[:, 1] 
            h = R_top_right[:, 3] 
            w = R_top_right[:, 2] 
            
            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1
            
            x = R_bottom_left[:, 0] 
            y = R_bottom_left[:, 1] 
            h = R_bottom_left[:, 3] 
            w = R_bottom_left[:, 2] 

            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1
            
            x = R_bottom_right[:, 0] 
            y = R_bottom_right[:, 1] 
            h = R_bottom_right[:, 3] 
            w = R_bottom_right[:, 2] 

            idx = 0
            for not_used in x:
                cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=mask_color, thickness=cv2.FILLED   )
                idx +=1

        else:
            reflected_img = img
            
        '''
        Drawing on the original non reflected portion.  
        '''
        idx = 0
        for not_used in x:
            x0 = x[idx]
            y0 = y[idx]
            x1 = x[idx] + w[idx]
            y1 = y[idx] + h[idx]

            cv2.rectangle(reflected_img, (x0,y0), (x1,y1), color=mask_color, thickness=cv2.FILLED)
            idx +=1
        ######### DRAWING RED MASKS ON BBOXES FOR REFLECTED IMAGE ###########
        
        #cv2.imshow("reflected", reflected_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        file_name = os.path.basename(image_id_to_filepath[img_id]) 

        save_path_for_image = str(reflected_imgs_save_path) + "/bbox_filled_" + str(file_name) 

        # will be deleted at the end of processing.
        #self.temp_filepaths.append(save_path_for_image)

        #print(save_path_for_image)
        cv2.imwrite(save_path_for_image, reflected_img)

        return save_path_for_image, mask_color, img


    def _get_reflected_bounding_boxes(self, bboxcoords, img, padding):
        """
        This method gets all of the bounding boxes within the padded portions of the image. This method
        only works for reflected padding. 

        Parameters
        ----------

        bboxcoords : numpy array 
             A numpy array containing all of the bounding boxes in the format [[x, y, width, height]]
            
        img : numpy array
             This is the image matrix needed.  Use a numpy generated image, skimage read, or opencv read image.
            
        padding : int 
             Padding value is used to get the bounding boxes that would show up in the padded region.  So if you have 
             reflected your image this value is how this method will find the bounding boxes.
            

        Returns
        -------

        numpy array : right_reflected_bboxes
             The bounding boxes associated with the right-most padding

        numpy array : left_reflected_bboxes
             The bounding boxes associated with the left-most padding

        numpy array : top_reflected_bboxes
             The bounding boxes associated with the top-most padding

        numpy array : bottom_reflected_bboxes
             The bounding boxes associated with the bottom-most padding

        numpy array : top_left_bboxes
             The bounding boxes associated with the top-left-most padding

        numpy array : top_right_bboxes
             The bounding boxes associated with the top-right-most padding

        numpy array : bottom_left_bboxes
             The bounding boxes associated with the bottom-left-most padding

        numpy array : bottom_right_bboxes
             The bounding boxes associated with the bottom-right-most padding
        
        """
        #bboxes must be in x,y,w,h format

        # there will be 8 total reflected bboxes.  The reflection distances are equal to the distance
        # of each point of the bbox from the edges of the original image.  
        # 1st get distances from the edges of the original image
        # 2nd apply the distances to the bboxes.  This will give us 4 of the reflected boxes
        # 3rd find the corner reflections

        img_height, img_width = img.shape[:2]
        # 1 distance per edge
        right_distances = abs(bboxcoords[:, 0] - img_width)      # how far from the right
        left_distances = abs(bboxcoords[:, 0])                   # the original x value is the distance from the left
        bottom_distances = abs(bboxcoords[:, 1] - img_height)    # how far from the bottom
        top_distances = abs(bboxcoords[:, 1])                    # the original y value is the distance from the top

        bboxcoords[:, 0] += padding
        bboxcoords[:, 1] += padding

        # now that I have the differences/distances I know that the differences are how far the points are in the reflected boundaries
        # so I take the points and move them to their respective edges by added or subtracting by the difference and then
        # I add or subtract another portion because that is how far they are in the reflected portion
        # It appears that the width and height need to be subtracted from the x and/or y values at certain reflections

        ##### RIGHT REFLECTION ##############
        right_reflected_bboxes = bboxcoords.copy()
        right_reflected_bboxes[:,0] = (right_reflected_bboxes[:,0] + right_distances) + right_distances
        right_reflected_bboxes[:, 0] -= right_reflected_bboxes[:,2] #subtracting width
        ##### RIGHT REFLECTION ##############

        ##### LEFT REFLECTION ##############
        left_reflected_bboxes = bboxcoords.copy()
        left_reflected_bboxes[:,0] = (left_reflected_bboxes[:,0] - left_distances) - left_distances
        left_reflected_bboxes[:, 0] -= left_reflected_bboxes[:,2] #subtracting width
        ##### LEFT REFLECTION ##############

        ##### TOP REFLECTION ##############
        top_reflected_bboxes = bboxcoords.copy()
        top_reflected_bboxes[:,1] = (top_reflected_bboxes[:,1] - top_distances) - top_distances
        top_reflected_bboxes[:,1] -= top_reflected_bboxes[:,3]
        ##### TOP REFLECTION ##############

        ##### BOTTOM REFLECTION ##############
        bottom_reflected_bboxes = bboxcoords.copy()
        bottom_reflected_bboxes[:,1] = (bottom_reflected_bboxes[:,1] + bottom_distances) + bottom_distances
        bottom_reflected_bboxes[:,1] -= bottom_reflected_bboxes[:,3]
        ##### BOTTOM REFLECTION ##############

        ##### TOP LEFT REFLECTION ##############
        top_left_bboxes = left_reflected_bboxes.copy()
        top_left_bboxes[:,1] = (top_left_bboxes[:,1] - top_distances) - top_distances
        top_left_bboxes[:,1] -= top_left_bboxes[:,3]
        ##### TOP LEFT REFLECTION ##############

        ##### TOP RIGHT REFLECTION ##############
        top_right_bboxes = right_reflected_bboxes.copy()
        top_right_bboxes[:,1] = (top_right_bboxes[:,1] - top_distances) - top_distances
        top_right_bboxes[:,1] -= top_right_bboxes[:,3]
        ##### TOP RIGHT REFLECTION ##############

        ##### BOTTOM LEFT REFLECTION ##############
        bottom_left_bboxes = bottom_reflected_bboxes.copy()
        bottom_left_bboxes[:,0] = (bottom_left_bboxes[:,0] -left_distances) -left_distances
        bottom_left_bboxes[:,0] -= bottom_left_bboxes[:,2]
        ##### BOTTOM LEFT REFLECTION ##############

        ##### BOTTOM RIGHT REFLECTION ##############
        bottom_right_bboxes = bottom_reflected_bboxes.copy()
        bottom_right_bboxes[:,0] = (bottom_right_bboxes[:,0] + right_distances) + right_distances
        bottom_right_bboxes[:,0] -= bottom_right_bboxes[:,2]
        ##### BOTTOM RIGHT REFLECTION ##############


        # Visualizer for the crops and bboxes.  To see anything from this method just replace the bboxes where
        # it is annotated to replace and then run the script
        """
        reflected_img = cv2.copyMakeBorder(img,padding,padding,padding,padding, cv2.BORDER_REFLECT)

        diameters = get_diameters(bboxcoords)

        patches = diameters * 5        

        x = bottom_right_bboxes[:, 0] #REPLACE WITH OTHER BBOXES HERE
        y = bottom_right_bboxes[:, 1] #REPLACE WITH OTHER BBOXES HERE
        h = bottom_right_bboxes[:, 3] #REPLACE WITH OTHER BBOXES HERE
        w = bottom_right_bboxes[:, 2] #REPLACE WITH OTHER BBOXES HERE

        midpoint_x = (x+x+w)//2
        midpoint_y = (y+y+h)//2

        offset = patches // 2

        starting_x = (midpoint_x - offset)
        ending_x = (midpoint_x + offset)
        starting_y = (midpoint_y - offset)
        ending_y = (midpoint_y + offset)

        #This is just to see if i was capturing the bounding box
        #cv2.rectangle(img, (x, y), ((x+w), (y+h)), color=(0,255,255), thickness=1)
        images = {}
        idx = 0
        for patch in patches:
            cv2.rectangle(reflected_img, (x[idx], y[idx]), ((x[idx]+w[idx]), (y[idx]+h[idx])), color=(255,255,0), thickness=1   )
            cropped_img = reflected_img[int(starting_y[idx]):int(ending_y[idx]), int(starting_x[idx]):int(ending_x[idx])]
            cropped_img = np.asarray(cropped_img)

            images[ idx ] = cropped_img
            
            cv2.imshow("test",cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            idx +=1
        """
        return right_reflected_bboxes, left_reflected_bboxes, top_reflected_bboxes, bottom_reflected_bboxes, top_left_bboxes, top_right_bboxes, bottom_left_bboxes, bottom_right_bboxes

    #This is correctly centering every cropped image
    def _get_padded_indices_of_bbox(self, img_path, bbox, padding): 
        """
        This method gets the padded indices of the bounding boxes supplied along.

        Parameters
        ----------
        img_path : string
             A path to the image.
            
        bbox : numpy array or 2 dimensional list
             A list of lists or a numpy array holding the bounding boxes.

        padding : int
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        Returns
        -------

        numpy array : numpy array 
             The first return value is named "x" these are the padded x values

        numpy array : numpy array 
             The second return value is named "y" these are the padded y values

        numpy array : numpy array 
             The third return value is named "w" these are the widths to the bounding boxes

        numpy array : numpy array 
             The fourth return value is named "h" these are the heights to the bounding boxes

        numpy array : numpy array 
             The fifth return value is named "img" this is the original image that was supplied from
             the image folder path set in the object.

        """
        #TODO: Change this method.  It doesn't need to be here and do unnecessary
        #calculations.
                                                                        

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        bbox = np.asarray(bbox)

        x = bbox[:, 0] + padding #The added portion is the same as the padding
        y = bbox[:, 1] + padding
        h = bbox[:, 3] 
        w = bbox[:, 2]

        #This is just to see if i was capturing the bounding box
        #cv2.rectangle(img, (x, y), ((x+w), (y+h)), color=(0,255,255), thickness=1)
        
        return x, y, w , h,  img

    def generate_negatives(self, save_directory=None, n_cores=1, padding=0, generation_method=INPAINT): 
        """
        This method generates negative examples using the generation_method selected.  This means
        that the images supplied will be used to generate new images without any postive examples on
        the returned images.  This can be done via inpainting only as of right now
        
        TODO: Make more methods to generate negative examples

        Parameters
        ----------
        img_path : string
             (Default value = None)
             A path to the image directory
            
        n_cores : int
             (Default value = 1)
             The number of cores that should be used for generation.  Normally higher means faster, but more RAM used.

        padding : int
             (Default value = 0)
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        generation_method : int
             (Default value = INPAINT)
             This is a value that determines the type of algorithm that will be used to generate the negative examples.
             This is also part of the enums that are associated with the class so since one will most likely be using this
             outside of the class.  

        Example
        -------
        .. code-block:: python    

            from cvjson.cvjson.cvj import CVJ
            from cvjson.extensions.painter import Painter

            json_path = "path/to/your/json"

            cvj_obj = CVJ(json_path)
            painter_obj = Painter(cvj_obj)

            # Since we imported the Painter class from the painter module we can use the class variable INPAINT with Painter.INPAINT
            painter_obj.generate_negative_backgrounds(self, save_directory='.', n_cores=3, padding=3000, generation_method=Painter.INPAINT): 

        Returns
        -------

        list: list
             The only return value is named "negative_filepaths".  This value is a list of all the filepaths to 
             the generated images that are negative sample types.
        """

        # UNTIL AN UPDATE TO FIX THE CHECKPOINT METHODS OCCUR CLEARING THEM BEFORE RUNNING THE ALGORITHM WILL
        # ALWAYS OCCUR
        self.clear_checkpoints()
	
        if save_directory == None:
            save_directory = self.image_folder_path

        # This is found in the _reflected_inpaint and we need the directory name that the images
        # are saved in so we can move the files and remove the temp directory.
        temp_working_path = os.path.join(save_directory, "temp")

        if self.image_folder_path == None:
            print("Supply the image folder path to the object before calling this method.\n\n")
            return

        if generation_method == Painter.INPAINT:
            self.load_checkpoint()
            self.__generation_method_INPAINT(save_directory, n_cores, padding=padding, inplant_orignal=False, sample_type=Painter.NEGATIVE)
            
            ## creating negative category id based on phone keypad and the letters in negative 63428483
            # self._json_data["categories"].append(CVJ.NEGATIVE_CLASS)
    
            self.load_checkpoint()
            self.update_images(self.negative_filepaths)
            self.save_internal_json()
            
            if save_directory == None:
                self.negative_filepaths = set(self.negative_filepaths)
                print("Images are being saved at {}".format(self.image_folder_path))
                for file_ in list(self.negative_filepaths):
                    name = os.path.basename(file_)
                    new_path = os.path.join(self.image_folder_path, name)
                    os.rename(file_, new_path)
            else:
                 for file_ in list(self.negative_filepaths):
                    name = os.path.basename(file_)
                    new_path = os.path.join(save_directory, name)
                    os.rename(file_, new_path)

        self.update_checkpoint()
        shutil.rmtree(temp_working_path)

        return list(self.negative_filepaths)
        
    def generate_postives(self, save_directory=None, n_cores=1, padding=0, generation_method=INPAINT): 
        """
        This method generates negative examples using the generation_method selected.  This means
        that the images supplied will be used to generate new images without any postive examples on
        the returned images.  This can be done via inpainting only as of right now
        
        TODO: Make more methods to generate negative examples

        Parameters
        ----------
        img_path : string
             (Default value = None)
             A path to the image directory
            
        n_cores : int
             (Default value = 1)
             The number of cores that should be used for generation.  Normally higher means faster, but more RAM used.

        padding : int
             (Default value = 0)
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        generation_method : int
             (Default value = INPAINT)
             This is a value that determines the type of algorithm that will be used to generate the negative examples.
             This is also part of the enums/class variable that are associated with the class so since one will most likely be using this
             outside of the class.  

        Example
        -------
        .. code-block:: python    

            from cvjson.cvjson.cvj import CVJ
            from cvjson.extensions.painter import Painter

            json_path = "path/to/your/json"

            cvj_obj = CVJ(json_path)
            painter_obj = Painter(cvj_obj)

            # Since we imported the Painter class from the painter module we can use the class variable INPAINT with Painter.INPAINT
            painter_obj.generate_negative_backgrounds(self, save_directory='.', n_cores=3, padding=3000, generation_method=Painter.INPAINT): 

        Returns
        -------

        list: list
             The only return value is named "positive_filepaths".  This value is a list of all the filepaths to 
             the generated images that are positive sample types.
        """
    
        # UNTIL AN UPDATE TO FIX THE CHECKPOINT METHODS OCCUR CLEARING THEM BEFORE RUNNING THE ALGORITHM WILL
        # ALWAYS OCCUR
        self.clear_checkpoints()

        # This is found in the _reflected_inpaint and we need the directory name that the images
        # are saved in so we can move the files and remove the temp directory.
        temp_working_path = os.path.join(save_directory, "temp")


        if self.image_folder_path == None:
            print("Supply the image folder path to the object before calling this method.")

        if generation_method == Painter.INPAINT:
            
            self.load_checkpoint()
            self.__generation_method_INPAINT(save_directory, n_cores, padding=padding, inplant_orignal=True, sample_type=Painter.POSTIVE)
            self.update_images(self.positive_filepaths)
            self.save_internal_json()

            self.positive_filepaths = set(self.positive_filepaths)

        for file_ in list(self.positive_filepaths):
            name = os.path.basename(file_)
            dir_name = os.path.dirname(self.image_folder_path)
            new_path = os.path.join(dir_name, name)
            os.rename(file_, new_path)

        self.update_checkpoint()
        shutil.rmtree(temp_working_path)
        return list(self.positive_filepaths)


    def __generation_method_INPAINT(self, save_directory, n_cores, padding, inplant_orignal, sample_type):
        """
        This method is a generation method for multiscale patch based inpainting from GMIC.  This method is called from another
        method that drives generation methods and uses the Generator Enums.

        Parameters
        ----------
        save_directory : string
             The path to the source directory where all your files will be save.  folders will be created accordingly
            
        n_cores : int
             The number of cpus wanted to be used
            
        padding : int
             (Default value = 3000)
             How much padding there should be in the image.  This should be relatively high to not get out of bounds errors

        inplant_orignal : bool
             (Default value = True)
             A bool that, if true, puts the original image back in the center of the newly inpainted image.

        sample_type : int
             This variable is an enum of Painter.POSITIVE or Painter.NEGATIVE, 1 or 0.  This tells the the generation method 
             how to name the images.

        Returns
        -------

        """
        if sample_type == Painter.NEGATIVE:
            negative_manager = Manager()
            self.negative_filepaths = negative_manager.list()
        else:
            positive_manager = Manager()
            self.positive_filepaths = positive_manager.list()

        # TODO: add transf_one_patch from dataset_medifor_patches.py

        filename2imageid = self.get_filename_2_image_id()
        img_id_2_anns_dict = self.get_image_id_2_anns()

        if not os.path.isdir(save_directory):
            
            os.makedirs(save_directory)
        else:
            
            file_names_in_finished_folder = os.listdir(save_directory)
            print("The amount of items in the final saving directory is {}".format(len(file_names_in_finished_folder)))

            if sample_type == Painter.NEGATIVE:
                for filepath in self.negative_filepaths:
                    try:
                        name = os.path.basename(filepath)
                        file_names_in_finished_folder.remove(os.path.basename(name))
                    except:
                        print("couldn't find file {} in the temp folder during initial scan and checkpoint resume".format(name))
            else:
                for filepath in self.positive_filepaths:
                    try:
                        name = os.path.basename(filepath)
                        file_names_in_finished_folder.remove(os.path.basename(name))
                    except:
                        print("couldn't find file {} in the temp folder during initial scan and checkpoint resume".format(name))

            image_ids_in_finished_folder = []

            if len(file_names_in_finished_folder) > 0:
                for name in file_names_in_finished_folder:
                    try:
                        name_ = name.split('_')[1]
                        image_ids_in_finished_folder.append(filename2imageid[name_])
                    except:
                        #print("couldn't find image {}".format(name_))   
                        pass
                

                for img_id in image_ids_in_finished_folder:
                    try:
                        del img_id_2_anns_dict[img_id]
                        print("deleted image id {}".format(img_id))
                    except KeyError:
                        print("Tried to delete image id: {} because it found an image associated with that id in the completed pictures directory".format(img_id))

        # delete all annotations and ids for any image not found in the image folder
        images_in_image_path = os.listdir(self.image_folder_path)
        # some images in the image folder wont be in the json so a try except
        # is used.  Passing it because we just dont care about images not
        # in the json.
        for img_name in filename2imageid:
            if img_name not in images_in_image_path:
                try:
                    img_id = filename2imageid[img_name]
                    del img_id_2_anns_dict[img_id]
                except:
                    pass
                
        images_count = len(img_id_2_anns_dict.keys())
        print("The amount of images left to inpaint are {}".format(images_count))

        assert n_cores != 0, "Don't tell the computer to use zero CPUs, give a number to n_cores variable"
        if n_cores > images_count:
            n_cores = images_count
            if n_cores == 0:
                print("The program has detected that it has already completed the request.  Check the images in the saving directory. Or the images files supplied do not match the images within the json file.")
                return 
        elif n_cores > multiprocessing.cpu_count():
            n_cores = multiprocessing.cpu_count()

        

        pool = Pool(n_cores)
        pool.starmap(self._reflected_inpaint, zip(img_id_2_anns_dict, repeat(save_directory), repeat(padding), repeat(inplant_orignal), repeat(sample_type),
        repeat(self.negative_filepaths), repeat(self.positive_filepaths)))

    def remove_generated(self, sample_type):
        """
        
        Parameters
        ----------
        sample_type : int
            This is an enum/class variable associated with the Painter class.  This variable is used to determine what generated sample types
            will be removed from the image directory where the generated images are located.  Each generated image should have a beginning name
            start of "negative" or "positive"
        Returns
        -------
        list: list
            This is the list of filepaths of images that are being removed.
        """

        if sample_type == Painter.ALL:
            '''
            #Debug
            list_ = [os.path.join(self.image_folder_path, file) for file in os.listdir(self.image_folder_path) if file.split('_')[0] =="negative" or file.split('_') == "positive"]

            print(list_)
            '''
            image_list = os.listdir(self.image_folder_path)
            remove_list = [ file for file in image_list if file.split('_')[0] =="negative" or file.split('_')[0] == "positive"]
            NU = [os.remove(os.path.join(self.image_folder_path, file)) for file in image_list if file.split('_')[0] =="negative" or file.split('_') == "positive"]
            self.update_images(remove_list, True)

        elif sample_type == Painter.NEGATIVE:
            image_list = os.listdir(self.image_folder_path)
            remove_list = [ file for file in image_list if file.split('_')[0] =="negative"]
            NU = [os.remove(os.path.join(self.image_folder_path, file)) for file in image_list if file.split('_')[0] == "negative"]
            self.update_images(remove_list, True)
        
        elif sample_type == Painter.POSTIVE:
            image_list = os.listdir(self.image_folder_path)
            remove_list = [ file for file in image_list if file.split('_')[0] =="positive"]
            NU = [os.remove(os.path.join(self.image_folder_path, file)) for file in image_list if file.split('_')[0] == "positive"]
            self.update_images(remove_list, True)

        return remove_list




    # ALL CHECKPOINT METHODS CURRENTLY DO NOT RUN OUTSIDE OF THE INITIAL
    # PROCESS.  MEANING THEY ARE USELESS UNTIL I CAN FIGURE OUT HOW
    # TO NOT GET A FILE NOT FOUND ERROR IN MULTIPROCESSING AFTER THE
    # FILE IS FOUND USING OS.PATH.ISFILE().  AS OF RIGHT NOW THESE METHODS
    # ARE USELESS, BUT WILL BE KEPT IN THE CODE UNTIL AN UPDATE HAS OCCURRED
    def start_checkpoint(self):
        """

        NOTE
        ----
        This is not working as of right now, but is in the generation code.  This is one
        of the methods that will be corrected in the near future.

        This method starts the checkpoints by creating the checkpoints folder and
        creating empty pickle files.
        """

        checkpoints_folder = "checkpoints"

        if not os.path.isdir(checkpoints_folder):
            os.makedirs(checkpoints_folder)

        negative_checkpoint = "negative_checks.pkl"
        positive_checkpoint = "positive_checks.pkl"
        #temp_checkpoint = "temp_checks.pkl"

        negative_checkpoint = os.path.join(checkpoints_folder, negative_checkpoint)
        positive_checkpoint = os.path.join(checkpoints_folder, positive_checkpoint)
        
        if not os.path.isfile(negative_checkpoint):
            with open(negative_checkpoint, "wb") as file:
                pickle.dump(self.negative_filepaths, file)

        if not os.path.isfile(positive_checkpoint):
            with open(positive_checkpoint, "wb") as file:
                pickle.dump(self.positive_filepaths, file)

        # if not os.path.isfile(temp_checkpoint):
        #     with open(temp_checkpoint, "wb") as file:
        #         pickle.dump(self.temp_filepaths, file)


    def load_checkpoint(self):
        """

        NOTE
        ----
        This is not working as of right now, but is in the generation code.  This is one
        of the methods that will be corrected in the near future.

        This method loads the checkpoints which are pickled lists of
        filepaths for completed images or currently being worked on images.
        
        This is useful for sharing data between processes or if an interruption occurs during
        the generation, then the program can recover its position of what images are left
        to work on.
        """


        checkpoints_folder = "checkpoints"

        negative_checkpoint = "negative_checks.pkl"
        positive_checkpoint = "positive_checks.pkl"
        # temp_checkpoint = "temp_checks.pkl"

        negative_checkpoint = os.path.join(checkpoints_folder, negative_checkpoint)
        positive_checkpoint = os.path.join(checkpoints_folder, positive_checkpoint)

        if os.path.isfile(negative_checkpoint):
            with open(negative_checkpoint, "rb") as file:
                self.negative_filepaths = pickle.load(file) 

        if os.path.isfile(negative_checkpoint):
            with open(positive_checkpoint, "rb") as file:
                self.positive_filepaths = pickle.load(file) 

        # if os.path.isfile(negative_checkpoint):
        #     with open(temp_checkpoint, "rb") as file:
        #         self.temp_filepaths = pickle.load(file) 

        self.start_checkpoint()

    def update_checkpoint(self):
        """

        NOTE
        ----
        This is not working as of right now, but is in the generation code.  This is one
        of the methods that will be corrected in the near future.

        This method updates the checkpoints in the the checkpoints folder.
        These checkpoints are the lists of filepaths for images completed or
        currently being workedon by GMIC.
        
        """


        checkpoints_folder = "checkpoints"
        negative_checkpoint = "negative_checks.pkl"
        positive_checkpoint = "positive_checks.pkl"
        # temp_checkpoint = "temp_checks.pkl"

        negative_checkpoint = os.path.join(checkpoints_folder, negative_checkpoint)
        positive_checkpoint = os.path.join(checkpoints_folder, positive_checkpoint)
        
        print("Writing Checkpoint")
        with open(negative_checkpoint, "wb") as file:
            pickle.dump(self.negative_filepaths, file)


        with open(positive_checkpoint, "wb") as file:
            pickle.dump(self.positive_filepaths, file)

        # with open(temp_checkpoint, "wb") as file:
        #     pickle.dump(self.temp_filepaths, file)

    def clear_checkpoints(self):
        """
        This method clears out any pickled checkpoints found in the 
        checkpoints folder that is near where the script was calling a 
        generation method.
        
        """

        checkpoints_folder = "checkpoints"
        if not os.path.isdir(checkpoints_folder):
            return
        remove_list = []

        for file in os.listdir(checkpoints_folder):
            if file.endswith('.pkl'):
                remove_list.append(file)

        not_used = [os.remove(os.path.join(checkpoints_folder,file)) for file in remove_list]
