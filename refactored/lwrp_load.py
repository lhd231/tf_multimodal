import tensorflow as tf
import lrp

with tf.Session() as sess:
    name = "/export/mialab/users/nlewis/tf_multimodal/pretrain_models/outpu_model_fusion_08_06:16:06_fold:_0_batch:_20_hidden_3000_0.01"

    graph = tf.train.import_meta_graph(name + "/model.ckpt.meta")

    graph.restore(sess,name+'/model.ckpt')

    file_writer = tf.summary.FileWriter('/export/mialab/users/nlewis/tf_multimodal/pretrain_models/',sess.graph)
