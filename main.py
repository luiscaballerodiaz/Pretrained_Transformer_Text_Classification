from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import os


def get_tweet_content(list_paths):
  """
    Función para guardar en un diccionario el contenido de archivos txt
    que se introduce en su entrada.
    Devuelve un diccionario en el que las claves son el id del tweet, y
    el valor el texto del tweet.
  """
  output_dict = dict()
  for i in list_paths:
    tweet_id = i.split("/")[-1].split(".txt")[0]
    with open(i, 'r', encoding='utf8') as f:
        output_dict[int(tweet_id)] = f.read()

  return output_dict


def get_profner_data(profner_path_data):
    # Obtenemos el path a los txt de los tweets.
    path_to_txt = profner_path_data+"subtask-1/train-valid-txt-files/"
    tweets_train_files = [path_to_txt+"train/"+i for i in os.listdir(path_to_txt+"train/")]
    tweets_valid_files = [path_to_txt+"valid/"+i for i in os.listdir(path_to_txt+"valid/")]
    # Obtenemos diccionarios en los que el key es el tweet_id y el value el texto del tweet.
    train_txt_content = get_tweet_content(tweets_train_files)
    valid_txt_content = get_tweet_content(tweets_valid_files)

    # Cargamos dos dataframes con los tweet_id y la categoría de los tweets
    path_to_labeled = profner_path_data+"subtask-1/"
    train_tweets = pd.read_csv(path_to_labeled+"train.tsv",sep="\t")
    valid_tweets = pd.read_csv(path_to_labeled+"valid.tsv",sep="\t")

    # Introducimos a los df el campo de texto mapeando los diccionarios con tweet_id
    train_tweets["tweet_text"] = train_tweets['tweet_id'].map(train_txt_content)
    train_tweets["set"] = "train"
    valid_tweets["tweet_text"] = valid_tweets['tweet_id'].map(valid_txt_content)
    valid_tweets["set"] = "test"

    # Concatenamos el resultado
    output_df = pd.concat([train_tweets,valid_tweets],axis=0)
    # Eliminamos retorno de carro
    output_df["tweet_text"] = output_df.tweet_text.apply(lambda x: x.replace('\n', ' '))
    return output_df[["tweet_id","tweet_text","label","set"]].reset_index(drop=True)


# SETTINGS
max_seq_length = 48  # Note: from the previous token assessment, most tweets have less than 30 tokens
train_batch_size = 32
val_batch_size = 32
test_batch_size = 32
learning_rate = 2e-5
num_epochs = 3
model_name = 'BSC-LT/roberta-base-bne'
# Note: selected transformer model is from Hugging Face website (https://huggingface.co/BSC-LT/roberta-base-bne)

# MAIN CODE
# Generate dataframe from input data
df = get_profner_data(r'./profner/')
print(df.head(4))

texts = df.tweet_text.values  # Feature tweet_text is the transformer input
labels = df.label.values  # Feature label is the classification

# Split dataframe into train, validation and test sets (for comparison purposes test set is 6000-8000 samples)
train_texts = texts[:4000]
train_labels = labels[:4000]
val_texts = texts[4000:6000]
val_labels = labels[4000:6000]
test_texts = texts[6000:]
test_labels = labels[6000:]

# Transformer input is tokenized with the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts.tolist(), truncation=True, max_length=max_seq_length, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, max_length=max_seq_length, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, max_length=max_seq_length, padding=True)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

# Calculate steps
train_steps_per_epoch = int(len(train_dataset) / train_batch_size)
val_steps_per_epoch = int(len(val_dataset) / val_batch_size)

# Create batches for simulation shuffling data to ensure randomness
train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality(), seed=0).batch(train_batch_size)
val_dataset = val_dataset.shuffle(buffer_size=val_dataset.cardinality(), seed=0).batch(val_batch_size)
test_dataset = test_dataset.batch(test_batch_size)

# Pretrained model configuration using AutoConfig and TFAutoModel classes
# Note current application is Sequence Classification using tensorflow
config = AutoConfig.from_pretrained(model_name, num_labels=df.label.nunique())
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True, config=config)

# Compile the model with optimizer, loss and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Run the simulation
history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_dataset, validation_steps=val_steps_per_epoch)

# Verify the test set accuracy
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)
