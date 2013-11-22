powerspectrum
=============

Ghetto python spectrum analysis.

Here is me whistling:
![alt tag](https://raw.github.com/coryking/powerspectrum/master/img/whistle.png)

Here is me talking:
![alt tag](https://raw.github.com/coryking/powerspectrum/master/img/talking.png)

This project consists of a command line util "plot-spectrum", which takes a .wav file and outputs a pretty looking display.

How it works
------------
This thing works by:
 1. Loading the entire file into memory
 2. Splitting the sample data into small chunks, given by "samples-sec".
 3. Running numpy's Discrete Fourier Transform function on each chunk.
 4. Plotting each chunk using matplotlib

Note:  This code was done on a long weekend for the heck of it just to explore Fourier Transforms.  It may/may not work on your machine and probably has tons of bugs in it.  Lord knows the combination of sound files it works on.
