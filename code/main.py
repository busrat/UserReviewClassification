# !pip install transformers
# !pip install tensorflow

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense

# Import data
df = pd.read_csv('../dataset/app_reviews_all_annotated2.csv')
df = df[['review', 'argument_cat', 'decision_cat']]

# Remove missing rows
df = df.dropna()

# Convert to numeric for bert
df['Argument'] = pd.Categorical(df['argument_cat'])
df['Decision'] = pd.Categorical(df['decision_cat'])
df['argument_cat'] = df['Argument'].cat.codes
df['decision_cat'] = df['Decision'].cat.codes

# Split into training and testing 
df, df_test = train_test_split(df, test_size = 0.2, stratify = df[['decision_cat']])

# BERT model
modelName = 'bert-base-uncased'
maxLen = 100
conf = BertConfig.from_pretrained(modelName)
conf.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = modelName, config = conf)

# Load transformer BERT model
transformerModel = TFBertModel.from_pretrained(modelName, config = conf)

bert = transformerModel.layers[0]
inputIds = Input(shape=(maxLen,), name='input_ids', dtype='int32')
inputs = {'input_ids': inputIds}

bertModel = bert(inputs)[1]
dropout = Dropout(conf.hidden_dropout_prob, name='pooled_output')
pooledOutput = dropout(bertModel, training=False)

arg = Dense(units=len(df.argument_cat.value_counts()), kernel_initializer=TruncatedNormal(stddev=conf.initializer_range), name='argument')(pooledOutput)
dec = Dense(units=len(df.decision_cat.value_counts()), kernel_initializer=TruncatedNormal(stddev=conf.initializer_range), name='decision')(pooledOutput)
outputs = {'argument': arg, 'decision': dec}

# Show model
model = Model(inputs=inputs, outputs=outputs, name='BERT_For_App_Review_Classification')
model.summary()

# Training
optimizer = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
loss = {'argument': CategoricalCrossentropy(from_logits = True), 'decision': CategoricalCrossentropy(from_logits = True)}
metric = {'argument': CategoricalAccuracy('accuracy'), 'decision': CategoricalAccuracy('accuracy')}
model.compile(optimizer = optimizer, loss = loss, metrics = metric)

x = tokenizer(text=df.review.to_list(), add_special_tokens=True, max_length=maxLen, 
              truncation=True, padding=True, return_tensors='tf', return_token_type_ids = False,
              return_attention_mask = True, verbose = True)

history = model.fit(x={'input_ids': x['input_ids']}, y={'argument': to_categorical(df.argument_cat), 'decision': to_categorical(df.decision_cat)},
                    validation_split=0.2, batch_size=64, epochs=10)



# Evaluation
testArg = to_categorical(df_test.argument_cat, 4)
testDec = to_categorical(df_test.decision_cat)
testReview = tokenizer(text=df_test['review'].to_list(), add_special_tokens=True,
                         max_length=maxLen, truncation=True, padding=True, 
                         return_tensors='tf', return_token_type_ids = False,
                         return_attention_mask = False, verbose = True)
modelEval = model.evaluate(x={'input_ids': testReview['input_ids']}, y={'argument': testArg, 'decision': testDec})
