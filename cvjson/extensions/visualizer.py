from cvjson.cvj import CVJ    
import cv2
import numpy as np
import os

class Visualizer(CVJ):
    """
    The Visualizer class takes the json Handle
    and can perform checks and clean the images
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

    def visualize_bboxes(self, color=(255,45,55), timer=None):

        image_id_2_anns = self.get_image_id_2_anns()
        image_id_2_filepath = self.get_image_id_2_filepath()
        class_id_to_name = self.get_class_id_2_name()

        for id in image_id_2_anns:
            img = cv2.imread(image_id_2_filepath[id], cv2.IMREAD_COLOR)
            for ann in image_id_2_anns[id]:
                bbox = ann["bbox"]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(x2), int(y2)), color=(255,255,0), thickness=1)
                class_string = "Class = {}, Class ID= {}".format(class_id_to_name[ann["category_id"]], ann["category_id"])
                cv2.putText(img, class_string, (int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                print(class_string)
                print(image_id_2_filepath[id])

            if timer == None:
                cv2.imshow("visualizing", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.imshow("visualizing", img)
                cv2.waitKey(int(timer * 1000))
                cv2.destroyAllWindows()


    def generate_drawn_annotations(self, output_dir, color= (255,45,55)):

        image_id_2_anns = self.get_image_id_2_anns()
        image_id_2_filepath = self.get_image_id_2_filepath()
        class_id_to_name = self.get_class_id_2_name()

        for id in image_id_2_anns:
            img = cv2.imread(image_id_2_filepath[id], cv2.IMREAD_COLOR)
            for ann in image_id_2_anns[id]:
                bbox = ann["bbox"]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(x2), int(y2)), color=(255,255,0), thickness=1)
                class_string = "Class = {}, Class ID= {}".format(class_id_to_name[ann["category_id"]], ann["category_id"])
                cv2.putText(img, class_string, (int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                print(class_string)
                print(image_id_2_filepath[id])

                cv2.imwrite(os.path.join(output_dir, os.path.basename(image_id_2_filepath[ann["image_id"]])), img)




    def visualize_segmentation(self, alpha=0.3, color= (255,45,55)):

        image_id_2_anns = self.get_image_id_2_anns()
        image_id_2_filepath = self.get_image_id_2_filepath()


        for id in image_id_2_anns:
            img = cv2.imread(image_id_2_filepath[id], cv2.IMREAD_COLOR)
            overlay = img.copy()

            for ann in image_id_2_anns[id]:
                segm = ann["segmentation"]
                print(segm)
                poly_list = []
                
                for poly in segm:
                    index = 0
                    for coord in poly:
                        if index % 2 == 0:
                            x = coord
                            y = poly[index + 1]
                        
                        point = np.asarray([x, y])
                        poly_list.append(point)
                        index += 1


                    cv2.fillPoly(overlay, np.asarray([poly_list]),  1)
                    cv2.putText(overlay, "segmentation", (poly[0], poly[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


            cv2.imshow("visualizing", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


