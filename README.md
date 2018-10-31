# Time Stretch
This python script implements stretching or shrinking audio data without changing the pitch. Given an input wav file, it stretches(shrinks) it and saves to a new output file. The file can also be imported as a library function.
The audio is first convert to frequency domain resulting in a short-term fourier transform. The STFT is then transformed according to [The Phase Vocoder](https://labrosa.ee.columbia.edu/matlab/pvoc/) algorithm, and ultitmately converted back to time domain.
The functionality is inside `stretch()` function. I wrote a loop-less version of this function in `stretch_wo_loop` in order boost speed, but it just led to less a readable function without much change is speed.
The command line usage for running the script is:
```
usage: speed up or speed down the audio without changing the pitch
       [-h] -i IN_FILE [-o OUT_FILE] -f FACTOR [-n NFFT]

optional arguments:
  -h, --help            show this help message and exit
  -i IN_FILE, --in-file IN_FILE
                        path to the input wav file
  -o OUT_FILE, --out-file OUT_FILE
                        path of the output file
  -f FACTOR, --factor FACTOR
                        factor by which to shrink or dilate time. if FACTOR <
                        1.0, audio will be sped up, otherwise sped down
  -n NFFT, --nfft NFFT  num of FFT bins to use
```