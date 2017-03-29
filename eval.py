#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_dbrcnn_pos import TextDBRCNN
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = "concept5_test"
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "concept5_test":
        datasets = data_helpers.get_datasets_concept5(cfg["datasets"][dataset_name]["training_data_file"]["path"],
                                                  cfg["datasets"][dataset_name]["target_data_file"]["path"])
        datasets_pos = data_helpers.get_datasets_concept5(cfg["datasets"][dataset_name]["pos_training_data_file"]["path"],
                                                  cfg["datasets"][dataset_name]["pos_target_data_file"]["path"])

    x_text, y = data_helpers.load_data_labels(datasets)
    x_text_pos, y_pos = data_helpers.load_data_labels(datasets_pos)

    y_test = np.argmax(y, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        x_text = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        x_text = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

# Map data into vocabulary
l1 = [len(x.split(" ")) for x in x_text]
l2 = [len(x.split(" ")) for x in x_text_pos]
count = 0
for index in range(len(l1)):
    if l1[index] == l2[index]:
        count += 1
print("Accuracy: {:g}".format(count/float(len(l1))))

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_path = os.path.join(os.path.abspath(os.path.join(FLAGS.checkpoint_dir, os.pardir)), "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_text_pos)))
x_new_test = np.concatenate((datasets_pos['index'],x_test), axis=1).astype(np.int)


max_word_length = 10
x_char_text = list()
for stnc in x_text:
    words = stnc.split(' ')
    stnc_length = len(words)
    num_padding = max_document_length - stnc_length
    for word in words:
        x_char_text.append(' '.join(list(word)))
    for i in range(num_padding):
        x_char_text.append('')
x_char_text = np.array(x_char_text)
vocab_processor_char = learn.preprocessing.VocabularyProcessor(max_word_length)
x_char = np.array(list(vocab_processor_char.fit_transform(x_char_text)))
x_char_test = np.reshape(x_char,(-1,max_document_length*max_word_length))


print("\nEvaluating...\n")

def position(pos):
    max_length = max_document_length
    p = []
    for j in range(len(pos)):
        position_array = np.zeros(max_length)
        for i in range(len(position_array)):
            position_array[i] = i - pos[j] + max_length - 1
        p.append(position_array)
    return np.array(p)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
        input_position_1 = graph.get_operation_by_name("input_position_1").outputs[0]
        input_position_2 = graph.get_operation_by_name("input_position_2").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]


        # Generate batches for one epoch
        batches = data_helpers.batch_iter(
            list(zip(x_new_test[:,2:], x_char_test, position(x_new_test[:,0]), position(x_new_test[:,1]))), FLAGS.batch_size, 1, shuffle=False)


        # Generate batches for one epoch
        # batches = data_helpers.batch_iter(list(x_new), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for batch in batches:
            x_batch, x_char_batch, position_batch_1, position_batch_2 = zip(*batch)
            feed_dict = {
                input_x: x_batch,
                input_x_char: x_char_batch,
                input_position_1: position_batch_1,
                input_position_2: position_batch_2,
                dropout_keep_prob: 1.0
            }
            batch_predictions = sess.run(predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
x_text_array = np.array([(' '.join(x)).encode('utf-8') for x in np.concatenate((datasets['index'],[[x_] for x_ in x_text]), axis=1)])
predictions_human_readable = np.column_stack((x_text_array, all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
