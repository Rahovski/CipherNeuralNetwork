import numpy as np
import cv2
import pathlib
method = cv2.TM_SQDIFF_NORMED


pathToImage = "D:\Work\SHP\EuralNetworkTeach\CipherNeuralNetwork\WordCipher"
origRoot = pathlib.Path(pathToImage)
allImagesPath = list(origRoot.glob('*jpg'))
allImagesPath = [str(path) for path in allImagesPath]
print(allImagesPath)
AllSumPath = list(origRoot.glob('*png'))
AllSumPath = [str(path) for path in AllSumPath]
print(AllSumPath)

image = cv2.imread(allImagesPath[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread(AllSumPath[0])
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray2, gray, method)
