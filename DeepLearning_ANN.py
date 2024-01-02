 [markdown]
# # Deep learning ANN

Box_office_v3ANN = pd.read_csv('.data/Box_office_v3ANN.csv')
Box_office_v3ANN.head()


import tensorflow as tf
print ("TensorFlow version: " + tf.__version__)


classifier = tf.keras.Sequential()


# Adding the input layer and the first hidden layer
classifier.add(tf.keras.layers.Dense(6, activation='relu', kernel_initializer='uniform',input_dim=12949))


classifier.add(tf.keras.layers.Dense(4, activation='relu', kernel_initializer='uniform'))


classifier.add(tf.keras.layers.Dense(3, activation='relu', kernel_initializer='uniform'))


classifier.add(tf.keras.layers.Dense(2, activation='relu', kernel_initializer='uniform'))


classifier.add(tf.keras.layers.Dense(1, activation='relu', kernel_initializer='uniform'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

 [markdown]
# # ANN Prediction


training_target.shape


training_features.shape



classifier.fit(training_features, training_target, batch_size = 10, epochs=30)
# evaluate the model
scores = classifier.evaluate(training_features, training_target)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))


%%time

# Predicting the Test set results
ANN_pred = classifier.predict(test_features)
ANN_pred = (ANN_pred > 0.5)


# Confusion Matrix
CMANN = confusion_matrix(test_target,ANN_pred)
CMANN


# Accuracy Score
ACANN= accuracy_score(test_target, ANN_pred )
print(ACANN)


print(classifier.summary())


!pip install keras


from tensorflow import keras
import keras;


# import keras;
# from keras.models import Sequential;
# from keras.layers import Dense;

network = tf.keras.Sequential();
        #Hidden Layer#1
network.add(tf.keras.layers.Dense(units=6,
                  activation='relu',
                  kernel_initializer='uniform',
                  input_dim=11));

        #Hidden Layer#2
network.add(tf.keras.layers.Dense(units=6,
                  activation='relu',
                  kernel_initializer='uniform'));

        #Exit Layer
# network.add(Dense(units=1,
network.add(tf.keras.layers.Dense(units=1,
                  activation='sigmoid',
                  kernel_initializer='uniform'));




! pip install ann_visualizer


from ann_visualizer.visualize import ann_viz;

ann_viz(network, title="");

 [markdown]
# # Models Compared


# Accuracy Score

print(" ANN Prediction Accuracy                                : {:.2f}%".format(ACANN * 100))




















