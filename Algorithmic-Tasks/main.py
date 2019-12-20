import numpy as np
import tensorflow as tf
  
if __name__ == '__main__':
  import sys
  sys.path.append('../Utils')
  from tasks import addition as task # any task could be analyzed  
  from Active_Memory_Transformer import Active_Memory_Transformer, argument

  arg = argument()
  arg.vocab_size = arg.input_vocab_size = arg.target_vocab_size = 20
  arg.att = 'vanilla_self_attention'
  
  arg.unidirectional_encoder = False
  model = Active_Memory_Transformer(arg)
  
  sequence_size = 5
  batch_size = 32
  vocab_size = arg.vocab_size
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  epochs = 5
  iterations = 100
  
  should_continue = True
  epochs_since_update = 0
  epoch = 0

  while should_continue:
    epoch += 1
    train_cost = []
    for iteration in range(iterations):
      trainX, trainY = task(batch_size,
                            sequence_size,
                            vocab_size)
      _, cost = sess.run([model.train_op, model.cost],
                         feed_dict = {model.inputs: trainX,
                                      model.targets: trainY,
                                      model.training: True,
                                      model.learning_rate: 0.01,
                                      model.keep_prob: 0.9})
      assert not np.isnan(cost), 'Epoch {}, Iteration {}'.format(epoch,
                                                                 iteration + 1)
      train_cost.append(cost)
    print('Epoch {}, Average Training Loss {:.4f}, Final Training Loss {:.4f}'.format(epoch,
                                                                                      np.mean(train_cost),
                                                                                      cost))
    testX, testY = task(batch_size,
                        sequence_size,
                        vocab_size)
    test_accuracy, predict = sess.run([model.accuracy, model.predict],
                                      feed_dict = {model.inputs: testX,
                                                   model.targets: testY,
                                                   model.training: False,
                                                   model.keep_prob: 1.0})
    print(testY[0])
    print(predict[0])
    print('Accuracy {:.4f}'.format(test_accuracy))
    if test_accuracy == 1.0: 
      sequence_size += 2 # this is where you adjust the sequence size
      epochs_since_update = 0
    else:
      epochs_since_update += 1
      if epochs_since_update == 20:
        should_continue = False
        print('Maximum Sequence Size: {}, Epochs: {}'.format(sequence_size,
                                                             epoch))
    if epoch == 100:
      should_continue = False
      print('Maximum Sequence Size: {}, Epochs: {}'.format(sequence_size,
                                                           epoch))
      
  sess.close()