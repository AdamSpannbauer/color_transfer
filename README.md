Super fast color transfer between images
==============

The <code>color_transfer</code> package is an OpenCV and Python implementation based (loosely) on [*Color Transfer between Images*](http://www.thegooch.org/Publications/PDFs/ColorTransfer.pdf) [Reinhard et al., 2001] The algorithm itself is extremely efficient (much faster than histogram based methods), requiring only the mean and standard deviation of pixel intensities for each channel in the L\*a\*b\* color space.

For more information, along with a detailed code review, [take a look at this post on my blog](http://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/).

# Requirements
- OpenCV
- NumPy

# Install
To install, make sure you have installed NumPy and compiled OpenCV with Python bindings enabled.

From there, there easiest way to install is via pip:

`$ pip install color_transfer`

# Examples
Below are some examples showing how to run the <code>example.py</code> demo and the associated color transfers between images.

`$ python example.py --source images/autumn.jpg --target images/fallingwater.jpg`
![Autumn and Fallingwater screenshot](docs/images/autumn_fallingwater.png?raw=true)

`$ python example.py --source images/woods.jpg --target images/storm.jpg`
![Woods and Storm screenshot](docs/images/woods_storm.png?raw=true)

`$ python example.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg`
![Sunset and Ocean screenshot](docs/images/sunset_ocean.png?raw=true)

# `color_transfer` options & `auto_color_transfer `

The `color_transfer` function has additional arguments for toggling `clip` and `preserve_paper`.  These options can change how aesthetically pleasing the color transfer result is.  See `help(color_transfer.color_transfer)` for more info on the arguments (or view [this pull request thread](https://github.com/jrosebr1/color_transfer/pull/5) for a more in depth discussion)

A utility function, `auto_color_transfer`, has been added to easily view the results of all combinations of these 2 arguments.  In addition to showing all possible transfers, `auto_color_transfer ` suggests the transfer that is truest to the source image's color (using mean absolute error to compare HSV color channels).  Despite attempting to minimize error, the suggested result is not always the most aesthetically pleasing; this is why the comparison image is provided for the user to choose which result works best for the use case.

`$ python auto_transfer_example.py -s images/storm.jpg -t images/ocean_day.jpg`

<img src=docs/images/auto_transfer.png>

