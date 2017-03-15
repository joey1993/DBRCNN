import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, position_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_position_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_position_1")
        self.input_position_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_position_2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            print self.embedded_chars.get_shape()
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print self.embedded_chars_expanded.get_shape()
            self.embedded_chars_transpose = tf.transpose(self.embedded_chars,[1,0,2])
            self.embedded_chars_reshape = tf.reshape(self.embedded_chars_transpose,[-1,embedding_size])

        rnn_input = tf.split(0, sequence_length, self.embedded_chars_reshape)
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias = 1.0)
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias = 1.0)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias = 1.0)
        output, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input, dtype = tf.float32)
        output_pack = tf.transpose(tf.pack(output),[1,0,2])

        self.embedded_chars_expanded = tf.nn.avg_pool(tf.expand_dims(output_pack, -1), ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
        print self.embedded_chars_expanded.get_shape()


        with tf.device('/cpu:0'), tf.name_scope("position"):
            self.M = tf.Variable(tf.random_uniform([2*sequence_length-1, position_size], -1.0, 1.0),
                name="M")
            self.embedded_position_1 = tf.expand_dims(tf.nn.embedding_lookup(self.M, self.input_position_1),-1)
            self.embedded_position_2 = tf.expand_dims(tf.nn.embedding_lookup(self.M, self.input_position_2),-1)

            print self.embedded_position_1.get_shape()
            print self.embedded_position_2.get_shape()

            self.embedded_chars_expanded = tf.concat(2, [self.embedded_chars_expanded, self.embedded_position_1, self.embedded_position_2])


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size+2*position_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
