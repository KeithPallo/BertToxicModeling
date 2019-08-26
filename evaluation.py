import pandas as pd
import tensorflow as tf
import numpy as np

import os

import run_classifier
import tokenization
import modeling
import optimization

data = pd.read_csv("train.csv")
data = data.drop(['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual',
                  'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
                  'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim', 'other_disability', 
                  'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 
                  'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white', 'created_date', 
                  'publication_id', 'parent_id', 'article_id', 'rating', 'funny', 'wow', 'sad', 'likes', 'disagree', 
                  'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count'], axis=1)

# data['label'] = np.where(data['target']>=0.5,1,0)
# label_list = list(data['label'].unique())
label_list = [0,1,2,3,4,5,6,7,8,9,10]

training_frac = 0.8
train_len = int(len(data)*training_frac)
valid_len = int(len(data)*(1.0-training_frac))

train = data.iloc[:train_len, :]
valid = data.iloc[:valid_len, :]

BERT_CONFIG_FILE = "./bert/bert_config.json"
INIT_CHECKPOINT = './output4/model.ckpt-96567'
VOCAB_FILE = "./bert/vocab.txt"
OUTPUT_DIR = './output4/'
TRAIN_FILE = "./output4/train.tf_record"
VALID_FILE = "./output4/valid.tf_record"
DO_LOWER_CASE = False
DO_TRAIN = False
DO_EVAL = True
DO_PREDICT = False
USE_TPU = False
USE_ONE_HOT_EMBEDDING = False
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 28
EVAL_BATCH_SIZE = 28
PREDICT_BATCH_SIZE = 28
LEARNING_RATE = 1e-7 
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.0
MASTER = None
SAVE_CHECKPOINTS_STEPS = 5000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 1
TPU_CLUSTER_RESOLVER = None
IS_PER_HOST = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

# tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

# def convert_input(x):
#     return run_classifier.InputExample(guid=None, 
#                                        text_a = x["comment_text"],
#                                        text_b = None, 
#                                        label = x["label"])

# train_InputExamples = train.apply(convert_input, axis= 1)
# valid_InputExamples = valid.apply(convert_input, axis= 1)

# run_classifier.file_based_convert_examples_to_features(valid_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer, VALID_FILE)

NUM_TRAIN_STEPS = 0 # int(len(train_InputExamples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
NUM_WARMUP_STEPS = 0 # int(NUM_TRAIN_STEPS * WARMUP_PROPORTION)

valid_input_fn = run_classifier.file_based_input_fn_builder(
    input_file=VALID_FILE, 
    seq_length=MAX_SEQ_LENGTH, 
    is_training=False, 
    drop_remainder=False)

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

tpu_config = tf.contrib.tpu.TPUConfig(
    iterations_per_loop=ITERATIONS_PER_LOOP,
    num_shards=NUM_TPU_CORES,
    per_host_input_for_training=IS_PER_HOST)

run_config = tf.contrib.tpu.RunConfig(
    cluster=TPU_CLUSTER_RESOLVER,
    master=MASTER,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tpu_config)

model_fn = run_classifier.model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=NUM_TRAIN_STEPS,
    num_warmup_steps=NUM_WARMUP_STEPS,
    use_tpu=USE_TPU,
    use_one_hot_embeddings=USE_ONE_HOT_EMBEDDING)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE) 

estimator.evaluate(input_fn=valid_input_fn, steps=None)