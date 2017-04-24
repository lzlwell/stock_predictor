from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10

from mpi4py import MPI



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../train_shared/test_0_1/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model""")


tf.app.flags.DEFINE_string('classes', '0_1',
                           """Two super classes used for building models""")
tf.app.flags.DEFINE_integer('n_models', 2,
                            """Train two models at the same time""")
tf.app.flags.DEFINE_integer('n_async', 50,
                            """Number of steps before communication """)
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use""")
tf.app.flags.DEFINE_integer('process_offset', 0,
                            """in case multiple values are set""")
tf.app.flags.DEFINE_integer('num_threads', 3,
                            """Number of threads used, num_threads = num_models + 1""")
tf.app.flags.DEFINE_float('conv_reg_param', 0.9,
                          """sharing rate of the conv layers""")
tf.app.flags.DEFINE_float('fc_reg_param', 0.1,
                          """sharing rate of the fc layers""")

def main(argv=None):
    NETWORK_NAME = 'cifar-shared-' + str(FLAGS.n_models) + '-' + str(FLAGS.n_async) + '-' \
                   + str(FLAGS.conv_reg_param) + '-' + str(FLAGS.fc_reg_param)

    tf.set_random_seed(0)

    process_offset = FLAGS.process_offset
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # set random seed
    tf.set_random_seed(0)

    class_splits = [int(x) for x in FLAGS.classes.split('_')]


    if rank > 0:
        # build model for each agent
        print('start thread %d ...' % rank)
        ind = rank - 1
        cid = class_splits[ind]

        if FLAGS.n_models > FLAGS.num_gpus:
            gid = ind % FLAGS.num_gpus
else:
    gid = ind

        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(gid)):
                global_step = tf.Variable(0, trainable=False)

                # Get images and labels super class cid
                images, labels = cifar10.distorted_inputs(cid=cid)

                # build a graph that computes the logits predictions
                # from the inference model and return centroid assign ops
                logits, centroid_variables, restore_dict, restore_ops = cifar10.inference(images)

                # calculate loss
                loss = cifar10.loss(logits, labels)


                # Build a Graph that trains the model with one batch of examples and
                # updates the model parameters.
                train_op = cifar10.train(loss, global_step)

                # Create a saver.
                saver = tf.train.Saver(tf.all_variables())

                # Build the summary operation based on the TF collection of Summaries.
                summary_op = tf.merge_all_summaries()

                # Build an initialization operation to run below.
                #init = tf.global_variables_initializer()
                init = tf.initialize_all_variables()

                # Start running operations on the Graph.
                sess = tf.Session(config=tf_config)
                sess.run(init)

                # Start the queue runners.
                tf.train.start_queue_runners(sess=sess)

                save_path = os.path.join(FLAGS.train_dir, NETWORK_NAME+'/'+str(cid))

                if not tf.gfile.Exists(save_path):
                    tf.gfile.MakeDirs(save_path)

                summary_writer = tf.train.SummaryWriter(save_path, sess.graph)
                print("set up the model of super class %d" % cid)


    if rank == 0:
        # parameter sever thread
        print('start thread 0 ...')

        def avg_weight_list(weight_list):
            avgs = []
            for i in xrange(len(weight_list[0])):
                avgs.append(np.mean(np.vstack([np.expand_dims(np.array(w[i]), axis=0)
                                               for w in weight_list]), axis=0))
                return avgs

        with tf.device('/cpu:0'):
            time_taken_placeholder = tf.placeholder('float', [1])
            time_taken_summary = tf.scalar_summary('Time take', time_taken_placeholder[0])

        session = tf.Session(config=tf_config)

        save_path = os.path.join(FLAGS.train_dir, 'ParamSever-'+ NETWORK_NAME)

        if not tf.gfile.Exists(save_path):
            tf.gfile.MakeDirs(save_path)

        time_summary_writer = tf.train.SummaryWriter(save_path)

        session.run(tf.initialize_all_variables())


    # start training ...
    step = 0
    shared_centroid_weights = None
    out_data = np.array([0.0]*1000000)

    while step * FLAGS.n_async < FLAGS.max_steps:
        if rank == 0:
            start_time = time.time()

        if step > 0: # centroid weights broadcasting...
            if rank == 0:
                # send the centroid weights to worker
                for i in xrange(FLAGS.n_models):
                    tag_id = 2*step*(process_offset+1)
                    req = comm.isend(shared_centroid_weights, dest=i+1, tag=tag_id)
                    #print('pending: sent message %d to worker %d' % (tag_id, (i+1)))
                    req.wait()
                    #print('server sent message %d to worker %d' % (tag_id, (i+1)))
else:
    tag_id = 2*step*(process_offset+1)
    req = comm.irecv(source=0, buf=out_data, tag=tag_id )
    #print('pending: worker %d received message % d from server' % (rank, tag_id))
    shared_centroid_weights = req.wait()
    #print('worker %d received message % d from server' % (rank, tag_id))



        if rank > 0: # local model updates
            # update the model
            steps_so_far = step * FLAGS.n_async
            with tf.device('/gpu:' + str(gid)):
                centroid_weights = cifar10.run_steps(
                    sess, FLAGS.n_async, train_op, loss,
                    centroid_variables, restore_dict, restore_ops,
                    FLAGS, steps_so_far, shared_centroid_weights)

            tag_id = 2*step*(process_offset+1)+1
            req = comm.isend(centroid_weights, dest=0, tag=tag_id)
            #print('pending: worker %d sent message %d to server' % (rank, tag_id))
            req.wait()
            #print('worker %d sent message %d to server' % (rank, tag_id))

            # write summary
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step*FLAGS.n_async)

            # save checkpoint
            checkpoint_path = os.path.join(save_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step*FLAGS.n_async)


        if rank == 0:
            all_centroid_weights = []
            tag_id = 2*step*(process_offset+1)+1
            for i in xrange(FLAGS.n_models):
                req = comm.irecv(source=i+1, buf=out_data, tag=tag_id)
                #print('pending: sever received message %d from worker %d' % (tag_id, (i+1)))
                all_centroid_weights.append(req.wait())
                #print('sever received message %d from worker %d' % (tag_id, (i+1)))

            time_taken_value = (time.time() - start_time) / float(FLAGS.n_async)

            time_taken_summary_eval = session.run(
                time_taken_summary,feed_dict={time_taken_placeholder: [time_taken_value]})
            time_summary_writer.add_summary(time_taken_summary_eval, step*FLAGS.n_async)
            print('time taken to receive all messages %.5f' % time_taken_value)


            if FLAGS.conv_reg_param != 0.0:
                avg_conv_weights = avg_weight_list([x['centroid_conv_weights']
                                                    for x in all_centroid_weights])
                avg_conv_biases = avg_weight_list([x['centroid_conv_biases']
                                                   for x in all_centroid_weights])
else:
    avg_conv_weights = None
    avg_conv_biases = None

            if FLAGS.fc_reg_param != 0.0:
                avg_fc_weights = avg_weight_list([x['centroid_fc_weights']
                                                  for x in all_centroid_weights])
                avg_fc_biases = avg_weight_list([x['centroid_fc_biases']
                                                 for x in all_centroid_weights])
else:
    avg_fc_weights = None
    avg_fc_biases = None

            shared_centroid_weights = {'centroid_conv_weights': avg_conv_weights,
                                       'centroid_conv_biases': avg_conv_biases,
                                       'centroid_fc_weights': avg_fc_weights,
                                       'centroid_fc_biases': avg_fc_biases}

            print("Iter: " + str(step * FLAGS.n_async))

        step += 1



if __name__ == '__main__':
    tf.app.run()
