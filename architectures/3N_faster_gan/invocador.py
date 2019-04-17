import os
import time

import subprocess
import sys
from collections import OrderedDict

import time
import xml.etree.ElementTree

def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)

i = 0

d = OrderedDict()
d["time"] = time.time()

cmd = ['nvidia-smi', '-q', '-x']
cmd_out = subprocess.check_output(cmd)
gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

util = gpu.find("utilization")
d["gpu_util"] = extract(util, "gpu_util", "%")

d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
d["mem_used_per"] = d["mem_used"] * 100 / 11171

if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
	msg = 'Idle'
else:
      msg = 'Busy'   
###
my_script = "python caller.py &"


##
if msg == 'Idle':

      time.sleep(60)
      
      # Second Sampling
      if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
	      msg = 'Idle'
      else:
            msg = 'Busy'   

      if msg == 'Idle':
            os.system(my_script)
            print("GPU is Idle; Script has been called")

if msg == 'Busy':
      print("GPU is Busy")