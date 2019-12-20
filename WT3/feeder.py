import numpy as np
import sentencepiece as spm

# module load sentencepiece/0.1.83-python3.6 

class WT3Feeder():
  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.file_length = {'train': 3658413,
                        'valid': 7832,
                        'test': 8985}
    self.file_iteration = {'train': 0,
                           'valid': 0,
                           'test': 0}
    self.files = {'train': open('/data/WT3.train.txt',
                                'r'),
                  'valid': open('/data3/WT3.valid.txt',
                                'r'),
                  'test': open('/data/WT3.test.txt',
                               'r')}
    self.sp = spm.SentencePieceProcessor()
    try:
      self.sp.load('/home/c3204522/Dataset/WT3/lambada.model')
    except:
      self.sp.load('/mnt/c/james/Programming/Dataset/WT3/lambada.model')
    self.vocab_size = 32001
    self.dataset = {'train': None,
                    'valid': None,
                    'test': None}
    
  def __call__(self, task = 'train'):
    assert task == 'train' or task == 'test' or task == 'valid'
    if self.dataset[task] == None:
      self.dataset[task] = []
      for _ in range(self.batch_size * 50):
        if _ % 50 == 0:
          self.dataset[task].append([])
        if self.file_iteration[task] == self.file_length[task]:
          self.recall_task(task)
        line = self.files[task].readline()
        if line == '':
          self.recall_task(task)
        elif line == '<new> \n':
          self.file_iteration[task] += 1
          line = self.files[task].readline()
        self.dataset[task][-1].append(line)
    data = []
    maxlen = -1
    for _ in range(self.batch_size):
      text = self.sp.encode_as_ids(self.dataset[task][_][0])
      del self.dataset[task][_][0]
      if maxlen < len(text):
        maxlen = len(text)
      data.append(text)
    np_data = np.zeros([self.batch_size, maxlen])
    np_data.fill(-1)
    for batch in range(self.batch_size):
      length = len(data[batch])
      np_data[batch,:length] = data[batch]
    np_data += 1
    if(self.dataset[task][_] == []):
      self.dataset[task] = None
    return np_data
        
  def recall_task(self, task):
    self.file_iteration[task] = 0
    self.files[task].close()
    self.files[task] = open('/home/c3204522/Dataset/WT3/WT3.{}.txt'.format(task),
                            'r')
    
  def change_batch_size(self, batch_size):
    self.batch_size = batch_size
    
  def translate(self, array):
    for data in array:
      translate_data = []
      for number in data:
        if number != 0.0:
          translate_data.append(int(number) - 1)
      print(self.sp.decode_ids(translate_data))
    
if __name__ == '__main__':
  feeder = WT3Feeder(batch_size = 4)
  sequence_size = []
  for i in range(10):
    data = feeder()
    print(feeder.translate([data[-1]]))
  exit()
  for i in range(100):
    feeder('valid')
  for i in range(100):
    feeder('test')