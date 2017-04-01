#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_brcnn import TextDBRCNN
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
tf.flags.DEFINE_integer("epoch_size", 1, "Epoch Size (default: 200)")
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
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "semeval2010_task8":
        datasets = data_helpers.get_datasets_concept5(cfg["datasets"][dataset_name]["testing_data_file"]["path"],
                                                  cfg["datasets"][dataset_name]["testing_target_data_file"]["path"])
    elif dataset_name == "concept5":
        datasets = data_helpers.get_datasets_concept5(cfg["datasets"][dataset_name]["training_data_file"]["path"],
                                                  cfg["datasets"][dataset_name]["target_data_file"]["path"])

    x_text, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

# Map data into vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 96
print "max_document_length:",max_document_length
vocab_path = os.path.join(os.path.abspath(os.path.join(FLAGS.checkpoint_dir, os.pardir)), "vocab")
#vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_text)))
x_new_test = np.concatenate((datasets['index'],x_test), axis=1).astype(np.int)


def position(pos):
    max_length = max_document_length
    p = []
    print "pos:"
    print len(pos)
    for j in range(len(pos)):
        position_array = np.zeros(max_length)
        for i in range(len(position_array)):
            position_array[i] = i - pos[j] + max_length - 1
        p.append(position_array)
    return np.array(p)

print("\nEvaluating...\n")

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
        input_position_1 = graph.get_operation_by_name("input_position_1").outputs[0]
        input_position_2 = graph.get_operation_by_name("input_position_2").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(
            list(zip(x_new_test[:,2:], position(x_new_test[:,0]), position(x_new_test[:,1]))), FLAGS.batch_size, FLAGS.epoch_size, shuffle=False)
        #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
	for batch in batches:
            x_batch, position_batch_1, position_batch_2 = zip(*batch)
            feed_dict = {
                input_x: x_batch,
                input_position_1: position_batch_1,
                input_position_2: position_batch_2,
                dropout_keep_prob: 1.0
            }
            batch_predictions = sess.run(predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        #for x_test_batch in batches:
            #batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            #all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
   # print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
x_text_array = np.array([(' '.join(x)).encode('utf-8') for x in np.concatenate((datasets['index'],[[x_] for x_ in x_text]), axis=1)])
predictions_human_readable = np.column_stack((x_text_array, all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
