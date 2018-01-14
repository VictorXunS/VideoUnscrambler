# VideoUnscrambler

A python unscrambler for short videos in which frames have been randomly shuffled


### Prerequisites

```
pip install tsp_solver2
pip install imagehash
```


## How to use

1. Modify parameters.py according to your preferences

2. Put the corrupted video file inside a folder named project_root/Data/VideoName/video_name.ext

3. Run the script. Example:
```
python unscramble_video.py corrupted_video.mp4
```

The output video will be saved at project_root/Output/video_name.ext
Some computation files will be generated and put inside the folder project_root/Data/VideoName/

## A note on parameters

This option automatically archives previous computations files and allows to compute from scratch.
If set at False and previous computation files exist, they will be used instead of recomputed.
```
cfg.ARCHIVE_PREVIOUS_RUN (bool)
```

This option allows to further shuffle input data
In particular, it allows to test the algorithm with uncorrupted videos.
```
cfg.SHUFFLE_INPUT_FRAMES (bool)
```

## Authors

* **Victor Suo** - *Initial work* - [VideoUnscrambler](https://github.com/VictorXunS/VideoUnscrambler)


## Acknowledgments

OpenCV Optical flow tutorial (https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html)
ImageHash library (https://pypi.python.org/pypi/ImageHash)
Travelling salesman greedy algorithm (https://github.com/dmishin/tsp-solver)