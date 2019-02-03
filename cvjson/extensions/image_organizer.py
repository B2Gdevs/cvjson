from cvjson.cvj import CVJ   
import cv2
import os
from tqdm import tqdm
import subprocess
 
class Image_Organizer(CVJ):
    """ """
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

    def move_images(self, list_of_images=None, new_image_path=None, from_this=False, transfer=False, set_new_path=False):
        """
        This method just moves the images that are associated with the json
        to the newly supplied path.  The path must be a directory, however if the 
        directory does not exist then it will make one for you.  The object then 
        sets the image_folder_path attribute to the new location of the images.

        Parameters
        ----------
        list_of_images : list
             (Default = None)
              This is a list of image filepaths just in case there is a need to move images in a list to a new place.

        new_image_path : string
             (Default = None)
            The directory to move the images from or to
        
        from_this : bool
             (Default = False)
              This tells this method if the images should be coming from the image directory supplied using cvj_obj.image_folder_path = "path/to/images".
              Or if it is coming to the image location supplied with cvj_obj.image_folder_path.

        transfer : bool
             (Default = False)
              This tells the method if there is a transfer of images from the list_of_images variable and have nothing to do with the
              internal json data and tranfers them to the new location supplied with new_image_path.

        set_new_path : bool
             (Default = False)
              This variable is used to set the image folder path of this object instance to the new image folder path that has been supplied.
            

        Returns
        -------
        string : The supplied directory path.

        """

        if self.image_folder_path != None:

            if from_this == True: 
            
                subprocess.call(['mkdir', '-p', new_image_path])

                image_id_2_filepath = self.get_image_id_2_filepath()

                for image_id in tqdm(image_id_2_filepath):
                    old_file_path = image_id_2_filepath[image_id]
                    file_name = os.path.basename(old_file_path)
                    new_file_path = os.path.join(new_image_path, file_name)
                    os.rename(old_file_path, new_file_path)

                if set_new_path:
                    self.image_folder_path = new_image_path

            else:
                for root, directs, files in os.walk(new_image_path):
                    for file in files:
                        if file.endswith((".png", ".tif", ".jpg", "jpeg")):
                            os.rename(os.path.join(root, file), os.path.join(self.image_folder_path, file))

        elif list_of_images != None:

            if from_this == False: 
                for img_path in list_of_images:
                    name = os.path.basename(img_path)
                    os.rename(img_path, os.path.join(self.image_folder_path, name))

            elif transfer:
                if new_image_path != None:
                    for img_path in list_of_images:
                        name = os.path.basename(img_path)
                        os.rename(img_path, os.path.join(new_image_path, name))

        return new_image_path


    def partition_images(self, K):
        """
        This method partitions the images in to K subfolders in the image directory that
        has already been supplied to the object.  This does not partition them in any specific order
        is just rotates through the new folders it creates and places an image in each one until
        it is done.  This is mainly used when the folder size is too big to view or has some other weird
        behaviours when there are too many images consolidated.
        
        See "Introduction to the Image Organizer" for more details on setting the folder path.
        Just remember the image_folder_path attribute must be set.

        Parameters
        ----------
        K : int
            The number of folders the images should be partitioned in to.
            

        Returns
        -------

        """

        assert os.path.isfile(self._json_path), self._json_path
        assert os.path.isdir(self._image_folder_path), self._image_folder_path
        assert isinstance(K, int), str(K)

        json_name = os.path.basename(self._json_path)

        new_folder_paths = []
        for i in range(int(K)): #goes from 0 to K-1
            folder_name = json_name + "_partition_" + str(i+1)
            new_folder = os.path.join(self.image_folder_path, folder_name)
            new_folder_paths.append(new_folder)
            if not os.path.isdir(new_folder):
                os.makedirs(new_folder)

        i = 0
        for file in tqdm(os.listdir(self.image_folder_path)):
            if file.endswith((".png", ".jpg", ".jpeg", ".tif")):
                old_file_path = os.path.join(self.image_folder_path,file)
                new_file_path = os.path.join(new_folder_paths[i], file)
                os.rename(old_file_path, new_file_path)
                i += 1
                if i >= K:
                    i = 0

        print("Partitioning in folder {} is compelete".format(self._image_folder_path))

    def combine_subfolders(self):
        """ 
        This method combines the subfolders created by partition_images() in to one
        big folder of images.  Though this method should be able to work on any source folder
        as long as there are only images in the subfolders.

        This method is primarily used when the training algorithm can only hand a source directory of images
        and cannot walk the subfolders.
        """

        dir_list = next(os.walk(self._image_folder_path))[1]
        i = 0
        for dir in dir_list:
            dir_list[i] = os.path.join(self._image_folder_path, dir)
            i+= 1
        for root, directs, files in os.walk(self._image_folder_path):  

            for file in tqdm(files):
                if file.endswith((".png", ".jpg", ".jpeg", ".tif")):
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(self._image_folder_path, file)
                    os.rename(old_file_path, new_file_path)

        for dir in dir_list:
            os.removedirs(dir)

        print("Completed combining subfolders in to root image directory {}.".format(self._image_folder_path))
    
    def gather_negatives_to_internal_json(self, negative_image_path=None, move_images=False):
        """
        This method is used to gather any negatives generated from the Painter class
        and store them in the internal json data of this object. If the move_images varibale 
        is True then it will also move the images that were tranferred to the image folder path
        that is set inside the object.  Not the negative_image_path that has been supplied to this
        method.  If the negative_image_path is none then this object will look in the image folder path.
        
        Parameters
        ----------
        negative_image_path : string
            (Default = None)
            This is the path to a image folder that has negative sample type images created by the 
            Painter class.
                
        move_images : bool
            (Defaul = False)
             The variable is used to tell the object if the images that are in the negative image folder path
             should be moved to the image folder path that has been set with cvj_obj.image_folder_path = your/actual/images/to/use
        
        Returns
        -------

        """

        if negative_image_path == None:
            negative_image_path = self.image_folder_path
            move_images = False

        paths = []
        names = []
        for root, directs, files in os.walk(negative_image_path):
            for file in files:
                if file.split('_')[0] == "negative":
                    names.append(file)
                    paths.append(os.path.join(root, file))

        self.update_images(paths)

        if move_images:
            self.move_images(list_of_images=paths,from_this=False)

    def convert_images(self, convert_to=".png", save_internal=False):
        """
        This method converts the images found in the image folder path that has been 
        supplied to the CVJ object that this method is being called from. This method uses
        opencv to handle the conversion.
        
        Parameters
        ----------
        convert_to : string
            (Default = .png)
             This is the image type that this method will convert the images to.

        save_internal = bool
             (Default = False)
             This will save the internal json data to the supplied json path to the object calling this method if this is true.
             Refer to "Introduction to the CVJ" for more information about supplying a json path to the CVJ object

        Returns
        -------

        
        """

        print("Converting Images to {}".format(convert_to))
        for root, directs, files in os.walk(self.image_folder_path):
            for file in tqdm(files):
                name_without_extension = file.split('.')[0]
                new_name = name_without_extension + convert_to
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_COLOR)

                new_path = os.path.join(root, new_name)
                cv2.imwrite(new_path, img)

        self.replace_extensions_of_json_images(save=save_internal)



if __name__ == "__main__":
    print("dont run this as a main script")
