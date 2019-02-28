# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import np_utils
#
#
# class NeuralNetwork:
#     def __init__(self, input_nn, output=2, n_hidden_layers=1, n_neurons=None):
#         model = Sequential()
#         model.add(Dense(n_neurons[0], input_dim=input_nn, activation='relu'))
#         for i in range(1, n_hidden_layers):
#             model.add(Dense(n_neurons[i], activation='relu'))
#         model.add(Dense(output, activation='softmax'))
#         model.compile(loss='categorical_crossentropy', optimizer='adam')
#         self.model = model
#
#     def fit(self, train, labels):
#         self.model.fit(train, np_utils.to_categorical(labels), epochs=100, verbose=0)
#
#     def predict(self, test):
#         labels = self.model.predict(test, verbose=0)
#         (_, y) = labels.shape
#         predict = []
#         for label in labels:
#             max_value = max(label)
#             for i in range(y):
#                 if label[i] == max_value:
#                     predict.append(i)
#
#         return predict
#
#
