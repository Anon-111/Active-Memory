import numpy as np
import tensorflow as tf
import functools

import util_code as utils
import loss
import optimize
import develop_bias

class Active_Memory_Transformer():
  def __init__(self, arg,
               name = None):
    if name:
      self.name = name
    else:
      self.name = 'Active-Memory-Transformer'
    batch_size = 64
    input_sequence_size = 10
    output_sequence_size = 10
    if __name__ != '__main__':
      batch_size = input_sequence_size = output_sequence_size = None
    self.arg = arg
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, input_sequence_size],
                                 name = 'inputs')
    if self.arg.classification:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size],
                                    name = 'targets')
    else:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size, output_sequence_size],
                                    name = 'targets')
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    
    self.cost = 0.0
    
    if not self.arg.classification:
      self.target_sequence_size = tf.shape(self.targets)[1]
    
    self.encoder_self_attention_bias = develop_bias._create_mask(self.input_sequence_size,
                                                                 self.arg.unidirectional_encoder)
    if not self.arg.classification:
      self.encoder_decoder_attention_bias = tf.zeros([1, 1, self.target_sequence_size, self.input_sequence_size],
                                                     name = 'encoder_self_attention_bias')
      self.decoder_self_attention_bias = develop_bias._create_mask(self.target_sequence_size,
                                                                   self.arg.unidirectional_decoder)
    
    if self.arg.mask_loss:
      if self.arg.classification:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size],
                                        name = 'loss_mask')
      else:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size, output_sequence_size],
                                        name = 'loss_mask')
    else:
      self.loss_mask = None
    
    if self.arg.ffd == 'transformer_ffd':
      self.ffd = self.transformer_ffd
    elif self.arg.ffd == 'sru':
      from SRU import SRU
      self.ffd = SRU
    elif self.arg.ffd == 'sepconv':
      self.ffd = self.sepconv
      
    if self.arg.att == 'vanilla_self_attention':
      self.att = functools.partial(utils.multihead_attention,
                                   memory = None,
                                   total_key_depth = self.arg.head_size * self.arg.num_heads,
                                   total_value_depth = self.arg.head_size * self.arg.num_heads,
                                   output_depth = self.arg.hidden_size,
                                   num_heads = self.arg.num_heads,
                                   deparameterize = False,
                                   dropout_keep_prob = self.keep_prob,
                                   dropout_type = self.arg.dropout_type,
                                   relative_attention = self.arg.relative_attention,
                                   max_relative_position = self.arg.max_relative_position,
                                   adaptive_mask = self.arg.adaptive_mask)
    elif self.arg.att == 'convolution':
      self.att = functools.partial(self.convolution,
                                   act_fn = utils.gelu) 
    elif self.arg.att == 'CGRU':
      self.att = self.cgru
    elif self.arg.att == 'highway_convolution':
      self.att = self.highway_convolution
    elif self.arg.att == 'add_convolution_vanilla' or self.arg.att == 'add_highway_convolution_vanilla' or self.arg.att == 'add_persistant_convolution_vanilla':
      self.att = self.add_conv_van
      if self.arg.att == 'add_persistant_convolution_vanilla':
        self.build_persistant_vectors()
    elif self.arg.att == 'persistant_convolution':
      self.att = functools.partial(self.persistant_conv,
                                   act_fn = utils.gelu)
      self.build_persistant_vectors()
    elif self.arg.att == 'repeat_highway':
      self.att = functools.partial(self.repeat_highway,
                                   act_fn = utils.gelu)
      self.build_highway_vectors()
    elif self.arg.att == 'multihead_conv':
      self.att = functools.partial(self.multihead_conv,
                                   act_fn = utils.gelu)
                                   
    if 'stop' in self.arg.pos:
      embedding_size = self.arg.hidden_size - 1
    else:
      embedding_size = self.arg.hidden_size
    with tf.variable_scope('encoder_embedding'):
      encoder_input, enc_params = utils.embedding(self.inputs,
                                                  model_dim = embedding_size,
                                                  vocab_size = self.arg.input_vocab_size,
                                                  name = 'encode')
    if not self.arg.classification:
      with tf.variable_scope('decoder_embedding'):
        decoder_input, dec_params = utils.embedding(self.targets,
                                                    model_dim = embedding_size,
                                                    vocab_size = self.arg.target_vocab_size,
                                                    name = 'decode')
      if self.arg.use_decoder:
        params = dec_params
        del enc_params
      else:
        params = enc_params
        del dec_params
    else:
      params = enc_params
    
    with tf.variable_scope('positional_encoding'):
      with tf.variable_scope('encoder'):
        encoder_input = self.timing_position(encoder_input)
      with tf.variable_scope('decoder'):
        if not self.arg.classification:
          decoder_input = self.timing_position(decoder_input)
    
    with tf.variable_scope('encoder'):
      encoder_input = self.dropout_fn(encoder_input)
      self.encoding = True
      self.decoding = False
      encoder_output = self.encoder(encoder_input,
                                    encoder_self_attention_bias = self.encoder_self_attention_bias)
      if self.arg.adaptive_mask:
        self.encoder_l0 = tf.reduce_sum(self.encoder_l0)
    if arg.use_decoder:
      with tf.variable_scope('decoder'):
        decoder_input = tf.pad(decoder_input,
                               paddings = [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input = self.dropout_fn(decoder_input)
        self.encoding = False
        self.decoding = True
        decoder_output = self.decoder(decoder_input,
                                      encoder_output,
                                      decoder_self_attention_bias = self.decoder_self_attention_bias,
                                      encoder_decoder_attention_bias = self.encoder_decoder_attention_bias)
        self.decoding = False
    if self.arg.classification:
      if self.arg.use_decoder:
        decoder_output = decoder_output[:,-1]
      else:
        encoder_output = encoder_output[:,-1]
    with tf.variable_scope('output'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.hidden_size, self.arg.target_vocab_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [self.arg.target_vocab_size],
                             dtype = tf.float32)
      if arg.use_decoder:
        self.logits = tf.tensordot(decoder_output,
                                   weights,
                                   axes = 1) + bias
      else:
        self.logits = tf.tensordot(encoder_output,
                                   weights,
                                   axes = 1) + bias
      self.loss_cl = loss.Loss(self.logits,
                               self.targets,
                               self.arg.loss,
                               vocab_size = self.arg.target_vocab_size,
                               label_smoothing = self.arg.label_smoothing)
      cost = self.loss_cl.loss
    if self.arg.mask_loss:
      self.cost += tf.reduce_mean(cost * self.loss_mask)
    else:
      self.cost += tf.reduce_mean(cost)
    if self.arg.adaptive_mask:
      self.cost += 0.0001 * self.encoder_l0
      if self.arg.use_decoder:
        self.decoder_l0 = tf.reduce_sum(self.decoder_l0)
        self.cost += 0.0001 * self.decoder_l0
    if self.arg.weight_decay_regularization:
      l2_loss = self.loss_cl.l2_loss(tf.trainable_variables())
      l2_loss *= self.arg.weight_decay_hyperparameter
      self.cost += l2_loss
    self.optimizer = optimize.Optimizer(arg,
                                        loss = self.cost,
                                        learning_rate = self.learning_rate)
    self.optimizer.accuracy(self.logits,
                            self.targets,
                            mask = self.loss_mask)
    self.train_op = self.optimizer.train_op
    self.predict = self.optimizer.predict
    self.correct_prediction = self.optimizer.correct_prediction
    self.accuracy = self.optimizer.accuracy
    self.optimizer.sequential_accuracy(self.logits,
                                       self.targets,
                                       mask = self.loss_mask)
    self.sequential_accuracy = self.optimizer.sequential_accuracy
    self.fetches = [encoder_input, encoder_output, self.logits]
      
  def encoder(self, inputs,
              encoder_self_attention_bias):
    x = inputs
    if self.arg.adaptive_mask:
      self.encoder_l0 = []
    for layer in range(1,
                       self.arg.encoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        x = self.timing_position(x)
        with tf.variable_scope('attention'):
          y = utils.layer_norm(x)
          y = self.att(query = y,
                       bias = encoder_self_attention_bias)
          if self.arg.adaptive_mask:
            self.encoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
          
    with tf.variable_scope('output'):
      return utils.layer_norm(x)
    
  def decoder(self, inputs,
              memory,
              decoder_self_attention_bias,
              encoder_decoder_attention_bias):
    x = inputs
    if self.arg.adaptive_mask:
      self.decoder_l0 = []
    for layer in range(1,
                       self.arg.decoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        with tf.variable_scope('self_attention'):
          y = utils.layer_norm(x)
          y = self.att(query = y,
                       bias = decoder_self_attention_bias)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('encoder_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = memory,
                                        bias = encoder_decoder_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = False,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
    return utils.layer_norm(x)
    
  def transformer_ffd(self, x):
    if self.arg.use_relu:
      act_fn = tf.nn.relu
    else:
      act_fn = utils.gelu 
    x = utils.dense(x,
                    output_dim = self.arg.filter_size,
                    use_bias = True,
                    name = 'ffd_1')
    x = act_fn(x)
    x = self.dropout_fn(x)
    with tf.variable_scope('ffd_output'):
      return utils.dense(x,
                         output_dim = self.arg.hidden_size,
                         use_bias = True,
                         name = 'ffd_2')
  
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
  
  def sepconv(self, x):
    output = utils.separable_convolution_2d(x,
                                            hidden_size = self.arg.filter_size,
                                            kernel_size = 3,
                                            name = 'conv1')
    if self.arg.use_relu:
      output = tf.nn.relu(output)
    else:
      output = utils.gelu(output)
    output = self.dropout_fn(output)
    return utils.separable_convolution_2d(output,
                                          hidden_size = self.arg.hidden_size,
                                          kernel_size = 5,
                                          name = 'conv2')
  
  def timing_position(self, inputs):
    sequence_size = tf.shape(inputs)[1]
    
    if self.arg.pos == 'timing':
      return inputs + utils.add_timing_signal_1d(sequence_size = sequence_size,
                                                 channels = self.arg.hidden_size)
    elif self.arg.pos == 'emb':
      return inputs + utils.add_positional_embedding(inputs,
                                                     max_length = self.arg.input_max_length, ###
                                                     hidden_size = self.arg.hidden_size,
                                                     input_sequence_size = sequence_size,
                                                     name = 'positional_embedding')
    elif self.arg.pos == 'linear_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      
      stop /= sequence_size
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.cast(tf.expand_dims(stop,
                                    axis = 2),
                     dtype = tf.float32)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'tanh_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.nn.tanh(gamma * stop/sequence_size) + 1 - tf.nn.tanh(gamma)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'exp_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.exp(gamma * (stop - sequence_size) / sequence_size)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    else:
      return inputs
  
  def persistant_conv(self, query,
                      bias,
                      act_fn = tf.identity,
                      input_size = None,
                      hidden_size = None):
    if input_size == None:
      input_size = self.arg.hidden_size
    if hidden_size == None:
      hidden_size = self.arg.hidden_size
      
    if self.encoding and self.arg.unidirectional_encoder:
      persistant_vector = tf.tile(self.persistant_vector,
                                  [self.batch_size, 1, 1])
      query = tf.concat([persistant_vector, query],
                        axis = 1)
    elif self.encoding and not self.arg.unidirectional_encoder:
      persistant_vector_one = tf.tile(self.persistant_vector[0],
                                      [self.batch_size, 1, 1])
      persistant_vector_two = tf.tile(self.persistant_vector[1],
                                      [self.batch_size, 1, 1])
      query = tf.concat([persistant_vector_one, query, persistant_vector_two],
                        axis = 1)
    elif self.decoding:
      persistant_vector = tf.tile(self.decoder_persistant_vector,
                                  [self.batch_size, 1, 1])
      query = tf.concat([persistant_vector, query],
                        axis = 1)
    with tf.variable_scope('convolution'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.kernel, input_size, hidden_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [hidden_size],
                             dtype = tf.float32)
      return act_fn(tf.nn.convolution(query,
                                      weights,
                                      padding = 'VALID') + bias)
    
  def convolution(self, query,
                  bias,
                  act_fn,
                  input_size = None,
                  hidden_size = None):
    if input_size == None:
      input_size = query.shape.as_list()[-1]
    if hidden_size == None:
      hidden_size = self.arg.hidden_size
    with tf.variable_scope('convolution'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.kernel, input_size, hidden_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [hidden_size],
                             dtype = tf.float32)
      if self.encoding and self.arg.unidirectional_encoder or self.decoding and self.arg.unidirectional_decoder:
        query = tf.concat([tf.zeros([self.batch_size, self.arg.kernel - 1, input_size]), query],
                        axis = 1)
        return act_fn(tf.nn.convolution(query,
                                        weights,
                                        padding = 'VALID') + bias)
      else:
        return act_fn(tf.nn.convolution(query,
                                        weights,
                                        padding = 'SAME') + bias)
    
  def highway_convolution(self, query,
                          bias):
    with tf.variable_scope('convolution_one'):
      a = self.convolution(query,
                           None,
                           tf.identity)
    with tf.variable_scope('convolution_two'):
      b = self.convolution(query,
                           None,
                           self.sigmoid_cutoff)
    return tf.multiply(a, b) + tf.multiply(query, 1 - b)
    
  def repeat_highway(self, query,
                     bias,
                     act_fn = tf.identity):
    for layer in range(1,
                       self.arg.kernel_layers + 1):
      x = tf.pad(query, 
                 [[0, 0], [self.arg.kernel - 1, 0], [0, 0]])
      weights = self.repeat_weights[layer]
      bias = self.repeat_bias[layer]
      x = act_fn(tf.nn.convolution(x,
                                   weights,
                                   padding = 'VALID') + bias)
      query += x
    return query
    
  def multihead_conv(self, query,
                     bias,
                     act_fn = tf.identity):
    query = tf.reshape(query,
                       shape = [self.batch_size, self.input_sequence_size, self.arg.num_heads, -1])
    with tf.variable_scope('multihead_conv'):
    
      hidden_size = self.arg.hidden_size // self.arg.num_heads
    
      kernel = tf.get_variable('kernel',
                               shape = [self.arg.kernel, self.arg.num_heads, hidden_size, hidden_size],
                               dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [hidden_size],
                             dtype = tf.float32)
      if self.encoding and self.arg.unidirectional_encoder or self.decoding and self.arg.unidirectional_decoder:
        query = tf.pad(query,
                       [[0, 0], [self.arg.kernel - 1, 0], [self.arg.num_heads // 2 - 1, self.arg.num_heads // 2], [0, 0]])
        query = act_fn(tf.nn.convolution(query,
                                         kernel,
                                         padding = 'VALID') + bias)
      else:
        query = act_fn(tf.nn.convolution(query,
                                         kernel,
                                         padding = 'SAME') + bias)
    return tf.reshape(query,
                      [self.batch_size, self.input_sequence_size, -1])
    
  def build_highway_vectors(self):
    self.repeat_weights = {}
    self.repeat_bias = {}
    for layer in range(1,
                       self.arg.kernel_layers + 1):
      self.repeat_weights[layer] = tf.get_variable('weights_{}'.format(layer),
                                                   shape = [self.arg.kernel, self.arg.hidden_size, self.arg.hidden_size],
                                                   dtype = tf.float32)
      self.repeat_bias[layer] = tf.get_variable('bias_{}'.format(layer),
                                                shape = [self.arg.hidden_size],
                                                dtype = tf.float32)
  
  def add_conv_van(self, query,
                   bias):
    with tf.variable_scope('convolution'):
      if self.arg.att == 'add_highway_convolution_vanilla':
        conv_att = self.highway_convolution(query,
                                            None)
      elif self.arg.att == 'add_convolution_vanilla':
        conv_att = self.convolution(query,
                                    None,
                                    act_fn = utils.gelu)
      elif self.arg.att == 'add_persistant_convolution_vanilla':
        conv_att = self.persistant_conv(query,
                                        None,
                                        act_fn = utils.gelu)
    with tf.variable_scope('self_attention'):
      att = functools.partial(utils.multihead_attention,
                              memory = None,
                              total_key_depth = self.arg.head_size * self.arg.num_heads,
                              total_value_depth = self.arg.head_size * self.arg.num_heads,
                              output_depth = self.arg.hidden_size,
                              num_heads = self.arg.num_heads,
                              deparameterize = False,
                              dropout_keep_prob = self.keep_prob,
                              dropout_type = self.arg.dropout_type,
                              relative_attention = self.arg.relative_attention,
                              max_relative_position = self.arg.max_relative_position,
                              adaptive_mask = self.arg.adaptive_mask)
      van_att = att(query,
                    bias = bias)
    return conv_att + van_att  
    
  def cgru(self, query,
           bias):
    with tf.variable_scope('update'):
      update_gate = self.convolution(query,
                                     None,
                                     self.sigmoid_cutoff)
    with tf.variable_scope('reset'):
      reset_gate = self.convolution(query,
                                    None,
                                    self.sigmoid_cutoff)
    with tf.variable_scope('output'):
      return tf.multiply(query,
                         update_gate) + tf.multiply(1 - update_gate,
                                                    self.convolution(tf.multiply(query,
                                                                                 reset_gate),
                                                                     None,
                                                                     tf.nn.tanh))
      
  def sigmoid_cutoff(self, state):
    if self.arg.sigmoid_act == 'hard_sigmoid':
      return tf.maximum(tf.minimum(1.2 * tf.sigmoid(state) - 0.1,
                                   1.0),
                        0.0)
    elif self.arg.sigmoid_act == 'squeezed_elu':
      return (utils.squeeze_elu(state) + 1) / 2
    
  def build_persistant_vectors(self):
    if self.arg.unidirectional_encoder:
      self.persistant_vector = tf.get_variable('persistant_vector',
                                               shape = [1, self.arg.kernel - 1, self.arg.hidden_size])
    else:
      kernel_size_one = (self.arg.kernel - 1) // 2
      kernel_size_two = self.arg.kernel - kernel_size_one - 1
      self.persistant_vector = [tf.get_variable('persistant_vector_one',
                                                shape = [1, kernel_size_one, self.arg.hidden_size]),
                                tf.get_variable('persistant_vector_two',
                                                shape = [1, kernel_size_two, self.arg.hidden_size])]
    if self.arg.use_decoder:
      self.decoder_persistant_vector = tf.get_variable('persistant_vector',
                                                       shape = [1, self.arg.kernel - 1, self.arg.hidden_size])
  
def argument():
  arg = optimize.argument()
  arg.att = 'multihead_conv' # 'vanilla_self_attention' 'CGRU' 'convolution' 'highway_convolution' 'add_convolution_vanilla' 'persistant_convolution' 'add_highway_convolution_vanilla' 'add_persistant_convolution_vanilla' 'repeat_highway' 'multihead_conv' 'dynamic-lightweight'
  arg.dropout_type = 'vanilla' # 'vanilla' 'alpha'
  arg.ffd = 'transformer_ffd' # 'transformer_ffd' 'sru' 'sepconv'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'timing' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  arg.sigmoid_act = 'hard_sigmoid' # 'hard_sigmoid' 'squeezed_elu'
  
  arg.decoder_layers = 2
  arg.encoder_layers = 4
  arg.filter_size = 512
  arg.head_size = 64
  arg.hidden_size = 128
  arg.input_max_length = 10
  arg.input_vocab_size = 1000
  arg.kernel = 7
  arg.kernel_layers = 3
  arg.label_smoothing = 1.0
  arg.max_relative_position = 100
  arg.num_heads = 8
  arg.target_max_length = 10
  arg.target_vocab_size = 1000
  arg.weight_decay_hyperparameter = 0.001
  
  arg.adaptive_mask = False
  arg.classification = False
  arg.deparameterize = False
  arg.mask_loss = False
  arg.relative_attention = False
  arg.unidirectional_decoder = True
  arg.unidirectional_encoder = True
  arg.use_decoder = False
  arg.use_mos = False
  arg.use_relu = True
  arg.weight_decay_regularization = False
  return arg
  
if __name__ == '__main__':
  arg = argument()
  
  model = Active_Memory_Transformer(arg)
