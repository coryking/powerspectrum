powerspectrum
=============

Ghetto python spectrum analysis.

This adds a command line util "plot-spectrum", which takes a wave file and outputs a pretty looking display.

How it works
------------
This thing works by:
1. Loading the entire file into memory
2. Splitting the sample data into small chunks, given by "samples-sec".
3. Running numpy's Discrete Fourier Transform function on each chunk.
