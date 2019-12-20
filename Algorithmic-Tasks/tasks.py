import numpy as np

def copy(batch_size,
         sequence_size,
         vocab_size):
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  return trainX, trainX

def reverse(batch_size, 
            sequence_size,
            vocab_size):
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  return trainX, np.flip(trainX,
                         axis = -1)

def duplicate(batch_size,
              sequence_size,
              vocab_size):
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  dup_vector = np.zeros(shape = [batch_size, sequence_size],
                        dtype = np.int32)
  return np.concatenate([trainX, dup_vector],
                        axis = -1), np.concatenate([trainX, trainX],
                                                   axis = -1)

def remember(batch_size,
             sequence_size,
             vocab_size):
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  dup_vector = np.zeros(shape = [batch_size, sequence_size],
                        dtype = np.int32)
  return np.concatenate([trainX, dup_vector],
                        axis = -1), np.concatenate([dup_vector, trainX],
                                                   axis = -1)

def sort(batch_size, 
         sequence_size,
         vocab_size):
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  return trainX, np.sort(trainX,
                         axis = -1)

def arg_sort(batch_size, 
             sequence_size,
             vocab_size): # vocab_size must be greater then or equal to sequence_size
  trainX = np.random.randint(low = 0,
                             high = vocab_size,
                             size = [batch_size, sequence_size],
                             dtype = np.int32)
  return trainX, np.argsort(trainX,
                            axis = -1)

def addition(batch_size,
             sequence_size,
             vocab_size = None):
  total_number = '0:0'+str(sequence_size)+'b'
  total_number = '{'+total_number+'}'
  input_number = '0:0'+str(int((sequence_size - 1)/2))+'b'
  input_number = '{'+input_number+'}'
  maximum_value = 2 ** int((sequence_size - 1)/2) - 1
  a = np.random.randint(low = 0,
                        high = maximum_value,
                        size = [batch_size])
  b = np.random.randint(low = 0,
                        high = maximum_value,
                        size = [batch_size])
  c = a + b
  trainX = []
  trainY = []
  for batch in range(batch_size):
    input_a = [int(x) for x in input_number.format(a[batch])]
    input_b = [int(x) for x in input_number.format(b[batch])]
    target_c = [int(x) for x in total_number.format(c[batch])]
    trainX.append(np.concatenate([input_a, [2], input_b],
                                 axis = 0))
    trainY.append(target_c)
  trainX = np.array(trainX)
  trainY = np.array(trainY)
  return trainX, trainY

def multiply(batch_size,
             sequence_size,
             vocab_size = None):
  total_number = '0:0'+str(sequence_size)+'b'
  total_number = '{'+total_number+'}'
  input_number = '0:0'+str(int((sequence_size - 1)/2))+'b'
  input_number = '{'+input_number+'}'
  maximum_value = 2 ** int(sequence_size - 1) - 1
  a = np.random.randint(low = 0,
                        high = int(np.sqrt(maximum_value)),
                        size = [batch_size])
  b = np.random.randint(low = 0,
                        high = int(np.sqrt(maximum_value)),
                        size = [batch_size])
  c = a * b
  trainX = []
  trainY = []
  for batch in range(batch_size):
    input_a = [int(x) for x in input_number.format(a[batch])]
    input_b = [int(x) for x in input_number.format(b[batch])]
    target_c = [int(x) for x in total_number.format(c[batch])]
    trainX.append(np.concatenate([input_a, [2], input_b],
                                 axis = 0))
    trainY.append(target_c)
  trainX = np.array(trainX)
  trainY = np.array(trainY)
  return trainX, trainY

def not_function(batch_size,
                 sequence_size,
                 vocab_size = None):
  trainX = np.random.randint(low = 0,
                             high = 2,
                             size = [batch_size, sequence_size])
  return trainX, (trainX - 1) * -1

def and_function(batch_size,
                 sequence_size, # sequence_size should be divisible by 4
                 vocab_size = None):
  trainX = np.random.randint(low = 0,
                             high = 2,
                             size = [batch_size, sequence_size])
  trainY = np.zeros(shape = [batch_size, sequence_size],
                    dtype = np.int32)
  for batch in range(batch_size):
    i = 0
    for sequence in range(sequence_size):
      if i == 0:
        i = 1
      elif i == 1:
        trainY[batch,sequence - 1] = (trainX[batch,sequence] and trainX[batch,sequence - 1])
        i = 0
  return trainX, trainY

def nand_function(batch_size, 
                  sequence_size, # should be divisible by 4
                  vocab_size = None):
  trainX, trainY = and_function(batch_size = batch_size,
                                sequence_size = sequence_size)
  return trainX, (trainY - 1) * -1

def or_function(batch_size,
                sequence_size,
                vocab_size = None):
  trainX = np.random.randint(low = 0,
                             high = 2,
                             size = [batch_size, sequence_size])
  trainY = np.zeros(shape = [batch_size, sequence_size],
                    dtype = np.int32)
  for batch in range(batch_size):
    i = 0
    for sequence in range(sequence_size):
      if i == 0:
        i = 1
      elif i == 1:
        trainY[batch,sequence - 1] = (trainX[batch,sequence] or trainX[batch,sequence - 1])
        i = 0
  return trainX, trainY

def nor_function(batch_size,
                 sequence_size,
                 vocab_size = None):
  trainX, trainY = or_function(batch_size = batch_size,
                               sequence_size = sequence_size)
  return trainX, (trainY - 1) * -1

def exor_function(batch_size,
                  sequence_size,
                  vocab_size = None):
  trainX = np.random.randint(low = 0,
                             high = 2,
                             size = [batch_size, sequence_size])
  trainY = np.zeros(shape = [batch_size, sequence_size],
                    dtype = np.int32)
  for batch in range(batch_size):
    i = 0
    for sequence in range(sequence_size):
      if i == 0:
        i = 1
      elif i == 1:
        trainY[batch,sequence - 1] = int(trainX[batch,sequence] != trainX[batch,sequence - 1])
        i = 0
  return trainX, trainY

def exnor_function(batch_size,
                   sequence_size,
                   vocab_size = None):
  trainX = np.random.randint(low = 0,
                             high = 2,
                             size = [batch_size, sequence_size])
  trainY = np.zeros(shape = [batch_size, sequence_size],
                    dtype = np.int32)
  for batch in range(batch_size):
    i = 0
    for sequence in range(sequence_size):
      if i == 0:
        i = 1
      elif i == 1:
        trainY[batch,sequence - 1] = int(trainX[batch,sequence] == trainX[batch,sequence - 1])
        i = 0
  return trainX, trainY