from tensorflow.keras import layers
from tensorflow.keras import losses

#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#!pip install --upgrade simple_elmo
#
#!wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json
#!wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
#
#!mv elmo_2x1024_128_2048cnn_1xhighway_options.json options.json

from simple_elmo import ElmoModel

model = ElmoModel()

#model.load('/content',max_batch_size=32)

# directory should include options.json and *.hdf5
# download from below links and put them in same folder
#!wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json
#!wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
dir = "some dir"
model.load(dir,max_batch_size=16)

df = pd.read_csv('../dataset/app_reviews_all_annotated2.csv')
df = df[['review', 'argument_cat', 'decision_cat']]

# Remove missing rows
df = df.dropna()

df = df.groupby('argument_cat').filter(lambda x : len(x) > 1)
df = df.groupby('decision_cat').filter(lambda x : len(x) > 1)

# Convert to numeric for bert
df['Argument'] = pd.Categorical(df['argument_cat'])
df['Decision'] = pd.Categorical(df['decision_cat'])
df['argument_cat'] = df['Argument'].cat.codes
df['decision_cat'] = df['Decision'].cat.codes

df2 = df.drop(columns=['Argument','Decision'])
df2

train_df, test_df = train_test_split(df2, test_size = 0.1, stratify=df2[['argument_cat']])

x_train = train_df['review']
y_train = train_df['argument_cat']
x_test = test_df['review']
y_test = test_df['argument_cat']

import pickle

PATH = '../'

try:
    x_train_embeddings = pickle.load(open(PATH + "x_train_embeddings.pickle", "rb" ))
except (OSError, IOError) as e:
    x_train_embeddings = model.get_elmo_vector_average(x_train.to_list())
    pickle.dump(x_train_embeddings, open(PATH + "x_train_embeddings.pickle",'wb'))

#vectors = model.get_elmo_vector_average(x_train.to_list()[0:128])
#x_train_embeddings = model.get_elmo_vector_average(x_train.to_list())

x_train_embeddings.shape

try:
    x_test_embeddings = pickle.load(open(PATH + "x_test_embeddings.pickle", "rb" ))
except (OSError, IOError) as e:
    x_test_embeddings = model.get_elmo_vector_average(x_test.to_list())
    pickle.dump(x_test_embeddings, open(PATH + "x_test_embeddings.pickle",'wb'))

#test_vectors = model.get_elmo_vector_average(x_test.to_list()[0:128])
#x_test_embeddings = model.get_elmo_vector_average(x_test.to_list())

inputs = tf.keras.Input(shape=256)
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(inputs)

arg = tf.keras.layers.Dense(4, activation=tf.nn.softmax, name='argument')(x)
dec = tf.keras.layers.Dense(5, activation=tf.nn.softmax, name='decision')(x)
outputs = {'argument': arg, 'decision': dec}

my_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ELMo_For_App_Review_Classification')


#arg = Dense(units=len(df.argument_cat.value_counts()), kernel_initializer=TruncatedNormal(stddev=conf.initializer_range), name='argument')(pooledOutput)
#dec = Dense(units=len(df.decision_cat.value_counts()), kernel_initializer=TruncatedNormal(stddev=conf.initializer_range), name='decision')(pooledOutput)
#outputs = {'argument': arg, 'decision': dec}

my_model.summary()

optimizer = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
loss = {'argument': CategoricalCrossentropy(from_logits = True), 'decision': CategoricalCrossentropy(from_logits = True)}
metric = {'argument': CategoricalAccuracy('accuracy'), 'decision': CategoricalAccuracy('accuracy')}
my_model.compile(optimizer = optimizer, loss = loss, metrics = metric)



history = my_model.fit(x=x_train_embeddings, y={'argument': to_categorical(train_df.argument_cat), 'decision': to_categorical(train_df.decision_cat)},
                    validation_split=0.1, batch_size=64, epochs=15)

#my_model.compile(optimizer='sgd',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

#epochs = 10
##history = my_model.fit(
##    vectors,
##    epochs=epochs,y=y_train[0:128])
#
#history = my_model.fit(
#    vectors,
#    epochs=epochs,y=y_train)

testArg = to_categorical(test_df['argument_cat'], 4)
testDec = to_categorical(test_df['decision_cat'])

modelEval = my_model.evaluate(x=x_test_embeddings,y={'argument': testArg, 'decision': testDec})

for x in zip(my_model.metrics_names,modelEval):
  print(x)

#loss, accuracy = my_model.evaluate(x=test_vectors,y=y_test[0:128])
#loss, accuracy = my_model.evaluate(x=x_test_embeddings,y=y_test)

#print("Test Loss: " + str(loss))
#print("Test Accuracy: " + str(accuracy))