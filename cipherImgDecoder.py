# Work with path in os
import pathlib
# Work with img and plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
import numpy as np
from random import shuffle

PathToTrainSum = "D:\Work\SHP\EuralNetworkTeach\CipherNeuralNetwork\ordinarySumbol\listTrainSymbol"
trainSumRoot = pathlib.Path(PathToTrainSum)
print(trainSumRoot)
listOfPathToSum = list(trainSumRoot.glob('*/*jpg'))
listOfPathToSum = [str(path) for path in listOfPathToSum]
print(listOfPathToSum)

# Получаем наименование директорий, где хранятся трен.изо
# Дальше идет составление словаря через перечисления
# для составления сопоставлений буква - цифра
articleNames = sorted(item.name for item in trainSumRoot.glob('*/') if item.is_dir())
print(articleNames)
articleNamesToIndex = dict((name, index) for index, name in enumerate(articleNames))
print(articleNamesToIndex)

# Делаем сопоставлений изображений - цифра от литерала
imgNumberFromLit = [articleNamesToIndex[pathlib.Path(path).parent.name]
                    for path in listOfPathToSum]

print(imgNumberFromLit)


def preprocess_image(image):
    # print(type(image))
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [70, 70])
    # Convert from RGB([x, y, 3]) to gray ([x, y, 1])
    image = tf.image.rgb_to_grayscale(image)
    # image /= 255.0  # normalize to [0,1] range
    image /= 255.0
    # After that we can resize matrix from [x, y, c] tp
    # [x, y]
    image = tf.reshape(image, [70, 70])
    image = np.asarray(image, dtype="float32")
    # print(type(image))
    # print(image.shape)
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


pathToTestSum = listOfPathToSum[33]
labelFromSum = articleNames[0]
plt.imshow(load_and_preprocess_image(pathToTestSum))
plt.grid(False)
plt.colorbar()
plt.xlabel([])
plt.title(labelFromSum)
plt.show()

# Convert sum Label to numpy array for Neural Model
imgNumberFromLit = np.asarray(imgNumberFromLit, dtype="uint8")
print(imgNumberFromLit)
print(type(imgNumberFromLit))
print(imgNumberFromLit.shape)

print("Convert and check train data to numpy array")
trainSumData = [None] * len(listOfPathToSum)
for i in range(len(listOfPathToSum)):
    trainSumData[i] = load_and_preprocess_image(listOfPathToSum[i])
print(type(trainSumData))
print(trainSumData[0].shape)
trainSumData = np.asarray(trainSumData, dtype="float32")
print("After convert")
print(type(trainSumData))
print(trainSumData[1].shape)

CipherNeuralModel = keras.Sequential([
    keras.layers.Flatten(input_shape=[70, 70]),
    keras.layers.Dense(980, activation='relu'),
    keras.layers.Dense(300, activation='sigmoid'),
    keras.layers.Dense(60, activation='sigmoid'),
    keras.layers.Dense(26, activation='softmax')
])

CipherNeuralModel.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

CipherNeuralModel.fit(trainSumData, imgNumberFromLit, epochs=3, steps_per_epoch=140)

checkDataDir = "D:\Work\SHP\EuralNetworkTeach\CipherNeuralNetwork\ordinarySumbol\CheckSymbol"
checkSumRoot = pathlib.Path(checkDataDir)
print(checkSumRoot)
listOfPathToCheck = list(checkSumRoot.glob('*jpg'))
listOfPathToCheck = [str(path) for path in listOfPathToCheck]
print(listOfPathToCheck)

pathToCheck = listOfPathToCheck[0]
conCheckImg = load_and_preprocess_image(pathToCheck)
plt.imshow(conCheckImg)
plt.grid(False)
plt.colorbar()
plt.xlabel([])
plt.show()

# add third dim for img and send to model.
# Network return matrix of predicticts
conCheckImg = (np.expand_dims(conCheckImg, 0))
print(conCheckImg.shape)
predictions_signle = CipherNeuralModel.predict(conCheckImg)
print(predictions_signle)
predictions_signle = list(predictions_signle[0])
print(predictions_signle)
print(articleNames[predictions_signle.index(max(predictions_signle))])
print(CipherNeuralModel.summary())

savePath = "D:\Work\SHP\EuralNetworkTeach\CipherNeuralNetwork\highModelNeural"
tf.saved_model.save(CipherNeuralModel, savePath)

# convert_model = tf.function(lambda x: CipherNeuralModel(x))
# concrete_func = convert_model.get_concrete_function(
#     tf.TensorSpec(CipherNeuralModel.inputs[0].shape,
#                   CipherNeuralModel.inputs[0].dtype))
# convertor = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# tflite_model = convertor.convert()


# export_model = tf.saved_model.load(savePath)
# concrete_func = export_model.signatures[
#     tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
# ]
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.allow_custom_ops = True
# converter = converter.convert()




