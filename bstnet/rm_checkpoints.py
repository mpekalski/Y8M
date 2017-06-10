from tensorflow import gfile
from tensorflow import flags
from tensorflow import app
import os
import re
import subprocess

FLAGS = flags.FLAGS

if __name__ == "__main__":
   flags.DEFINE_string("model_dir", "/home/m/youtube-8m/model/", "folder with models")
   flags.DEFINE_string("log_file", "/home/m/youtube-8m/rm_checkpoints.log", "file to store all executed commands")


def main(unused_argv):
    #files1 = gfile.Globa()
    dirs = os.listdir(FLAGS.model_dir)    
    for fld in dirs:
       checkpoints = gfile.Glob(os.path.join(FLAGS.model_dir,fld,"model.ckpt-*.index"))
       ckpt_list = []
       for ckpt in checkpoints:
         fnd = re.findall('(\d+)\.index', ckpt)
         if fnd is not None:
            ckpt_list.append(int(fnd[0]))
       file_path = os.path.join(FLAGS.model_dir, fld, "model.ckpt-")
       cmd = "rm {0}{1}.{2}"
       ckpt_max = max(ckpt_list)
       for ckpt in ckpt_list:
           if ckpt < ckpt_max:
              with open(FLAGS.log_file,'a') as f:
                 for ext in ["index","data-00000-of-00001","meta"]:              
                    with open(FLAGS.log_file,'a') as f:
                       subprocess.call(cmd.format(file_path, str(ckpt), ext), stdout=f, stderr=subprocess.STDOUT, shell=True)
                       f.write(cmd.format(file_path, str(ckpt), ext) + '\n')
                       print(cmd.format(file_path, str(ckpt), ext))
   

if __name__=="__main__":
   app.run()
