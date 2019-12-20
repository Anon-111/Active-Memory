import os
import sys
import time
import numpy as np
import tensorflow as tf

def build_and_run_model(model,
                        arg,
                        load_model = False):
  
  from optimize import warmup_learning_rate
  
  arg.dropout_type = 'vanilla'
  arg.ffd = 'transformer_ffd'
  arg.pos = 'timing'
  
  arg.encoder_layers = 8 # 6
  arg.head_size = 64
  arg.hidden_dim = 256
  arg.filter_size = 1024
  arg.input_vocab_size = 32001
  arg.num_heads = 8
  arg.rnn_encoder_layers = 2
  arg.target_vocab_size = 32001
  
  arg.classification = False
  arg.mask_loss = True
  arg.relative_attention = False
  arg.unidirectional = True
  arg.unidirectional_decoder = True
  arg.unidirectional_encoder = True
  arg.use_attention = True
  arg.use_decoder = False
  arg.use_mos = False
  arg.use_relu = False
  arg.weight_decay_regularization = False # True

  arg.hidden_size = arg.hidden_dim
  arg.layers = arg.encoder_layers
  arg.vocab_size = arg.input_vocab_size
  
  if model.__name__ == 'RNN':
    arg.cell = 'lstm'
    arg.hidden_dim = 256
    arg.layers = 2
    
  print('loading model')
  model = model(arg)
  
  lr = warmup_learning_rate(arg.hidden_size ** -0.5,
                            warmup = 10000)

  print(model.name)
  print(arg.att)
  print('Hidden size: {}'.format(arg.hidden_size))
  if model.name == 'RNN':
    print('Cell type: {}'.format(arg.cell))
    print('Layers: {}'.format(arg.layers))
    if model.name == 'RNN':
      print('Unidirectional')
  else:
    print('Layers: {}'.format(arg.encoder_layers))
    if arg.unidirectional_encoder:
      print('Unidirectional')
    else:
      assert not arg.use_decoder
      print('ERROR: BIDIRECTIONAL MODEL')
      sys.exit()
  print('Trainable Variables: {}'.format(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
  
  if not os.path.exists(model.name):
    os.mkdir(model.name)

  from feeder import WT3Feeder
  
  feeder = WT3Feeder(batch_size = 64)
  
  batch_size = 64
  
  sess = tf.Session()
  saver = tf.train.Saver()
  if load_model:
    restored_model = os.listdir(model.name)[0]
    saver.restore(sess,
                  os.path.join(model.name,
                               restored_model))
    if restored_model[12] == '_':
      print('Restored model at epoch {}'.format(int(restored[11])))
      current_epoch = int(restored[11])
    elif restored_model[13] == '_':
      print('Restored model at epoch {}'.format(int(restored[11:13])))
      current_epoch = int(restored[11:13])
    elif restored_model[14] == '_':
      print('Restored model at epoch {}'.format(int(restored[11:14])))
      current_epoch = int(restored[11:14])
  else:
    sess.run(tf.global_variables_initializer())
    current_epoch = 0
  
  max_epochs = 100
  for epoch in range(current_epoch,
                     max_epochs):
    loss_array = []
    accuracy_array = []
    memory = np.zeros([arg.encoder_layers, batch_size, 0, arg.hidden_size])
    for iteration in range(1000):
      text = feeder()
      while np.shape(text)[1] > 300:
        text = feeder()
      feed_dict = {model.inputs: text[:,:-1],
                   model.targets: text[:,1:],
                   model.training: True,
                   model.keep_prob: 0.9,
                   model.learning_rate: lr(),
                   model.loss_mask: np.where(text[:,:-1] == 0,
                                             0.0,
                                             1.0)}
      if model.name == 'Transformer-XL':
        feed_dict[model.memory] = memory
      _, loss, accuracy = sess.run([model.train_op,
                                    model.cost,
                                    model.accuracy],
                                   feed_dict = feed_dict)
      if model.name == 'Transformer-XL':
        memory = sess.run(model.new_mems,
                          feed_dict = feed_dict)
      loss_array.append(loss)
      accuracy_array.append(accuracy)
    print('Epoch {}, Train Loss {:.4f}, Train Accuracy {:.4f}'.format(epoch + 1,
                                                                      np.mean(loss_array),
                                                                      np.mean(accuracy_array)))
    loss_array = []
    accuracy_array = []
    memory = np.zeros([arg.encoder_layers, batch_size, 0, arg.hidden_size])
    for iteration in range(100):
      text = feeder('valid')
      while np.shape(text)[1] > 300:
        text = feeder('valid')
      feed_dict = {model.inputs: text[:,:-1],
                   model.targets: text[:,1:],
                   model.training: False,
                   model.keep_prob: 1.0,
                   model.loss_mask: np.where(text[:,:-1] == 0,
                                             0.0,
                                             1.0)}
      if model.name == 'Transformer-XL':
        feed_dict[model.memory] = memory
      loss, accuracy = sess.run([model.cost,
                                 model.accuracy],
                                feed_dict = feed_dict)
      if model.name == 'Transformer-XL':
        memory = sess.run(model.new_mems,
                          feed_dict = feed_dict)
      loss_array.append(loss)
      accuracy_array.append(accuracy)
    print('Valid Loss {:.4f}, Valid Accuracy {:.4f}'.format(np.mean(loss_array),
                                                            np.mean(accuracy_array)))
    print('')
  if (epoch + 1) % 100 == 0 or (epoch + 1) == max_epochs:
    loss_array = []
    accuracy_array = []
    for iteration in range(100):
      text = feeder('test')
      while np.shape(text)[1] > 300:
        text = feeder('test')
      feed_dict = {model.inputs: text[:,:-1],
                   model.targets: text[:,1:],
                   model.training: False,
                   model.keep_prob: 1.0,
                   model.loss_mask: np.where(text[:,:-1] == 0,
                                             0.0,
                                             1.0)}
      if model.name == 'Transformer-XL':
        feed_dict[model.memory] = memory
      loss, accuracy = sess.run([model.cost,
                                 model.accuracy],
                                feed_dict = feed_dict)
      loss_array.append(loss)
      accuracy_array.append(accuracy)
    print('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(np.mean(loss_array),
                                                          np.mean(accuracy_array)))
  for file in os.listdir(model.name):
    os.remove(os.path.join(model.name,
                           file))
  saver.save(sess,
             os.path.join(model.name,
                          'model_epoch{}_of_{}'.format(epoch + 1,
                                                       max_epochs)))
  
if __name__ == '__main__':
  sys.path.append('../Utils')
  from Active_Memory_Transformer import Active_Memory_Transformer as model, argument
  arg = argument()
  arg.att = 'vanilla_self_attention'
  arg.kernel = 20
  
  build_and_run_model(model,
                      arg)