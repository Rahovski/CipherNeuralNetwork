import cv2
import pathlib


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


pathToConvert = "A/А1.jpg"
print(pathToConvert)
img_raw = cv2.imread(pathToConvert)
print(img_raw.shape)


# for i in range(len(listOfPathToSum)):
#     count_slash = listOfPathToSum[i].count("\\")
#     listOfPathToSum[i].replace("\\", "/", count_slash)
#     img_raw = cv2.imread(listOfPathToSum[i])
#     height, width, channels = img_raw.shape
#     print(height, width, channels)
