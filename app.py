#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/11/02 

import sys
import queue
from argparse import ArgumentParser
from traceback import print_exc

import sounddevice as sd
import webrtcvad
import numpy as np
import matplotlib ; matplotlib.use('QtAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import softvc


def int_or_str(x:str):
  return int(x) if x.isdigit() else x


def monitor(args):

  def update_plot(frame, *fargs):
    nonlocal plotdata

    # fast-forward to stream head
    while True:
      try: data = q.get_nowait()
      except queue.Empty: break

      shift = len(data)
      plotdata = np.roll(plotdata, -shift, axis=0)
      plotdata[-shift:, :] = data

    # updape wavform
    for column, line in enumerate(lines):
      line.set_ydata(plotdata[:, column])

    # updape melspec
    if False:
      delta_f = (high - low) / (args.columns - 1)
      fftsize = math.ceil(samplerate / delta_f)
      low_bin = math.floor(low / delta_f)
      magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
      magnitude *= args.gain / fftsize

    return lines

  def callback(indata:np.ndarray, outdata:np.ndarray, n_frame:int, time, status:sd.CallbackFlags):
    if status: print(status, file=sys.stderr)
    
    # copy to plot thread
    q.put(indata.copy())

    # send to playback
    if args.mode == 'echo':
      outdata[:] = indata
    else:
      if False:
        length = int(args.blocksize/args.sr*1000)   # time in ms
        n_slices = length // 10                     # 10ms
        seg_size = len(indata) // n_slices
        indata_fp16 = indata.astype(np.float16)
        flag = [vad.is_speech(indata_fp16[i*seg_size:(i+1)*seg_size, :].tobytes(), args.sr) for i in range(n_slices)]
      
      if indata.max() > 0.3:
        print('voice detected')
        cvtdata = softvc.convert(indata)
        outdata[:] = cvtdata
      else:
        outdata[:] = indata

  ''' Figure '''
  q = queue.Queue()

  fig, ax = plt.subplots(num='RealTime Soft-VC')
  plotdata = np.zeros((int(args.window * args.sr / (1000 * 10)), 1))
  lines = ax.plot(plotdata)
  ax.axis((0, len(plotdata), -1, 1))
  ax.set_yticks([0])
  ax.yaxis.grid(True)
  ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  fig.tight_layout(pad=0)

  ''' VAD '''
  vad = webrtcvad.Vad(mode=3)

  ''' Main Loop '''
  try:
    if args.mode == 'vc':
      softvc.init(args.sr, args.sr)
    
    print('Start streaming...')
    animation = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    stream = sd.Stream(
        device=(args.device_in, args.device_out),
        samplerate=args.sr, 
        blocksize=args.blocksize,
        dtype=np.float32, 
        latency=args.latency,
        channels=1, 
        callback=callback)
    with stream:
      plt.show()
    
    #opt = input('Press <Enter> to exit').strip()
    #while opt:
    #  opt = input().strip()
  except KeyboardInterrupt: print('Exit on Ctrl+C')
  except: print_exc()


if __name__ == '__main__':
  MODES = ['vc', 'echo']

  parser = ArgumentParser()
  parser.add_argument('-L', '--list', action='store_true', help='list all audio devices')
  parser.add_argument('--default', action='store_true', help='use with -L, show default audio devices')
  parser.add_argument('-M', '--mode', choices=MODES, default=MODES[0], help='app running mode')
  parser.add_argument('-Di', '--device_in',  type=int_or_str,           help='output device (numeric ID or substring)')
  parser.add_argument('-Do', '--device_out', type=int_or_str,           help='output device (numeric ID or substring)')
  parser.add_argument('-w', '--window',      type=int,   default=16000, help='visible time slot (default: %(default)s ms)')
  parser.add_argument('-i', '--interval',    type=int,   default=30,    help='minimum time between plot updates (default: %(default)s ms)')
  parser.add_argument('-b', '--blocksize',   type=int,   default=1600,  help='block size (default: %(default)s)')
  parser.add_argument('--sr',                type=int,   default=16000, help='sample rate for stream')
  parser.add_argument('--latency',           type=float,                help='desired latency in seconds')
  args = parser.parse_args()

  assert args.blocksize > 0, 'blocksize must be positive'
  assert args.blocksize % 160 == 0, 'blocksize must align to 160 for compatible with soft-vc'

  if args.list:
    def _print_api(i, cfg):
      print(f'[{i}] {cfg["name"]}')
      print(f'   devices: {cfg["devices"]}')
      print(f'   default_input_device: {cfg["default_input_device"]}')
      print(f'   default_output_device: {cfg["default_output_device"]}')
    def _print_device(cfg):
      print(f'[{cfg["index"]}] {cfg["name"]}')
      print(f'   hostapi: {cfg["hostapi"]}') 
      print(f'   max_input_channels: {cfg["max_input_channels"]}') 
      print(f'   max_output_channels: {cfg["max_output_channels"]}') 
      print(f'   default_low_input_latency: {cfg["default_low_input_latency"]}')
      print(f'   default_low_output_latency: {cfg["default_low_output_latency"]}')
      print(f'   default_high_input_latency: {cfg["default_high_input_latency"]}')
      print(f'   default_high_output_latency: {cfg["default_high_output_latency"]}')
      print(f'   default_samplerate: {cfg["default_samplerate"]}')

    if args.default:
      print('========== [Default Audio APIs] ==========')
      i = sd.default.hostapi
      cfg = sd.query_hostapis(i)
      _print_api(i, cfg)
      print()

      i, o = sd.default.device
      if i != -1:
        print('========== [Default Audio Input Devices] ==========')
        _print_device(sd.query_devices(i, 'input'))
      if o != -1:
        print('========== [Default Audio Output Devices] ==========')
        _print_device(sd.query_devices(o, 'output'))
    else:
      print('========== [Audio Devices] ==========')
      for i, cfg in enumerate(sd.query_devices()):
        _print_device(cfg)
      print()
      print('========== [Audio APIs] ==========')
      for i, cfg in enumerate(sd.query_hostapis()):
        _print_api(i, cfg)

  else:
    monitor(args)
