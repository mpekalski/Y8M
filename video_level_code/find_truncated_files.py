import tensorflow as tf
import glob
import os
from my_utils import mylog

trfs = ['HS', 'P7', 'B5', 'T5', 'rH', 'yb', 'L5', 'Et', 'hF','QF','iv','E9','x5','ai','44']
trfiles = ["YouTube.Kaggle/input/frame_level_link/train/train"+x+".tfrecord" for x in trfs]

allFiles = glob.glob("YouTube.Kaggle/input/frame_level_link/validate/*.tfrecord") 
allFiles = glob.glob("YouTube.Kaggle/input/video_level/test/*.tfrecord") 
allFiles = glob.glob("YouTube.Kaggle/input/GENERATED_DATA/f2test/*.tfrecord") 
allFiles = glob.glob("YouTube.Kaggle/input/GENERATED_DATA/f2test/Atest-a.tfrecord") 


files = allFiles[0: len(allFiles)]
#files = trfiles
mylog("Number of files to review: {}.".format( len(files) ) )

file_cnt = 0 
i = 0
opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

num_examples_file = open('/tmp/num_examples_by_file.csv', 'w')
num_examples_file.write('filename,num_examples\n')

for filename in files:
  file_cnt = file_cnt + 1
  #print('checking %d/%d %s' % (cnt, filesSize, filename))
  if file_cnt%200 == 0:
    mylog("Checked {} files.".format(file_cnt))

  num_examples = 0
  try: 
    for example in tf.python_io.tf_record_iterator(filename, options=opts): 
    #for example in tf.python_io.tf_record_iterator(filename): 
      tf_example = tf.train.Example.FromString(example)
      num_examples += 1
  except :
    i += 1
    mylog("problematic file #{} out of {}: {}".format(i,  file_cnt, filename))
    #os.remove(filename)
  else:
    num_examples_file.write("{},{}\n".format(filename, num_examples))

num_examples_file.close()
mylog("Done.")
