import pandas as pd
import tensorflow as tf
import numpy as np

import os

import run_classifier
import tokenization
import modeling
import optimization

data = pd.read_csv("test.csv")

BERT_CONFIG_FILE = "./bert/bert_config.json"
INIT_CHECKPOINT = './output6/model.ckpt-30863'
VOCAB_FILE = "./bert/vocab.txt"
OUTPUT_DIR = './output6/'
TEST_FILE = "./output6/test.tf_record"
DO_LOWER_CASE = False
DO_TRAIN = False
DO_EVAL = False
DO_PREDICT = True
USE_TPU = False
USE_ONE_HOT_EMBEDDING = False
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 28
EVAL_BATCH_SIZE = 28
PREDICT_BATCH_SIZE = 28
LEARNING_RATE = 1e-7 
NUM_TRAIN_EPOCHS = 2.0
WARMUP_PROPORTION = 0.1
MASTER = None
SAVE_CHECKPOINTS_STEPS = 5000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = None
TPU_CLUSTER_RESOLVER = None
IS_PER_HOST = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

def convert_input(x):
    return run_classifier.InputExample(guid=x["id"], 
                                       text_a = x["comment_text"],
                                       text_b = None, 
                                       label = 0)

test_InputExamples = data.apply(convert_input, axis= 1)

run_classifier.file_based_convert_examples_to_features(test_InputExamples, [0,1], MAX_SEQ_LENGTH, tokenizer, TEST_FILE)

NUM_TRAIN_STEPS = 0 # int(len(test_InputExamples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
NUM_WARMUP_STEPS = 0 # int(NUM_TRAIN_STEPS * WARMUP_PROPORTION)

test_input_fn = run_classifier.file_based_input_fn_builder(
    input_file=TEST_FILE, 
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
    num_labels=2,
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

predictions = estimator.predict(test_input_fn)

probs0 = []
probs1 = []

for p in predictions:
    probs0.append(p["probabilities"][0])
    probs1.append(p["probabilities"][1])
    
data["prediction0"] = probs0
data["prediction1"] = probs1

submission = data.drop(["comment_text"], axis =1)

submission.to_csv('submission6.csv', index=False, header=True)