import cv2
import pathlib

pathToConvert = "D:\Work\SHP\EuralNetworkTeach\CipherNeuralNetwork\ordinarySumbol\listTrainSymbol\А"
trainSumRoot = pathlib.Path(pathToConvert)
print(trainSumRoot)
listOfPathToSum = list(trainSumRoot.glob('*jpg'))
listOfPathToSum = [str(path) for path in listOfPathToSum]
print(listOfPathToSum)
