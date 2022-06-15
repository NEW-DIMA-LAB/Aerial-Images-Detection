import cv2
import os

if __name__ == '__main__':
    image_path = "C:/Dataset/stanford_campus_dataset/videos/gates/video1/image/"
    for img_name in os.listdir(image_path):

        print(img_name)
        image = cv2.imread(os.path.join(image_path, img_name))
        rows, cols, cns = image.shape
        imagae_re = cv2.resize(image,(500,500))
        print(rows,cols)
        cv2.imshow("img",imagae_re)
        cv2.waitKey(0)