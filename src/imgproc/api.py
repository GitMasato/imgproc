"""api for image process functions.

Basic required arguments are list of movies or pictures (or directories where pictures are stored). if no movie or picture is given, function will not be processed (print error message and return False)

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

see usage of each api function below;

"""
from typing import List, Tuple
from imgproc import process


def animate(*, picture: List[str] = None, is_colored: bool = False, fps: float = 20.0):
  """api to animate pictures (note: keyword-only argument)

  Args:
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      fps (float, optional): fps of created movie. Defaults to 20.0.
  """
  if not picture:
    print("no picture is given!")
    return False

  parameter = process.AnimatingParameter(picture, is_colored, fps)
  image_process = process.ProcessExecution(process.AnimatingPicture(parameter))
  image_process.execute()


def binarize(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  thresholds: Tuple[int, int] = None,
):
  """api to binarize movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      thresholds (Tuple[int, int], optional): [low, high] threshold values to be used to binarize movie/picture. Low threshold must be smaller than high one. Defaults to None. If this variable is None, this will be selected using GUI window
  """
  if not movie and not picture:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie:
    parameter = process.BinarizingParameter(movie, thresholds)
    p.append(process.BinarizingPicture(parameter))
  if picture:
    parameter = process.BinarizingParameter(picture, thresholds)
    p.append(process.BinarizingPicture(parameter))
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def capture(
  *,
  movie: List[str] = None,
  is_colored: bool = False,
  times: Tuple[float, float, float] = None,
):
  """api to capture movies (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). Start must be smaller than stop, and difference between start and stop must be larger than step. Defaults to None. If this variable is None, this will be selected using GUI window
  """
  if not movie:
    print("no movie is given!")
    return False

  parameter = process.CapturingParameter(movie, is_colored, times)
  image_process = process.ProcessExecution(process.CapturingMovie(parameter))
  image_process.execute()


def crop(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  is_colored: bool = False,
  postisions: Tuple[int, int, int, int] = None,
):
  """api to crop movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      postisions (Tuple[int, int, int, int], optional): [x_1, y_1,x_2, y_2] two positions to crop movie/picture. position_1 must be smaller than position_2 Defaults to None. If this variable is None, this will be selected using GUI window
  """
  if not movie and not picture:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie:
    parameter = process.CroppingParameter(movie, is_colored, postisions)
    p.append(process.CroppingMovie(parameter))
  if picture:
    parameter = process.CroppingParameter(picture, is_colored, postisions)
    p.append(process.CroppingPicture(parameter))
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def hist_luminance(
  *, picture: List[str] = None, is_colored: bool = False,
):
  """api to create histgram of luminance (bgr or gray) of picture (note: keyword-only argument)

  Args:
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
  """
  if not picture:
    print("no picture is given!")
    return False

  parameter = process.CreatingLuminanceHistgramParameter(picture, is_colored)
  image_process = process.ProcessExecution(
    process.CreatingLuminanceHistgramPicture(parameter)
  )
  image_process.execute()


def resize(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  is_colored: bool = False,
  scales: Tuple[float, float] = (1.0, 1.0),
):
  """api to resize movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      scales (Tuple[float, float], optional): [x, y] ratios in each direction to scale movie/picture. Defaults to (1.0,1.0).
  """
  if not movie and not picture:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie:
    parameter = process.ResizingParameter(movie, is_colored, scales)
    p.append(process.ResizingMovie(parameter))
  if picture:
    parameter = process.ResizingParameter(picture, is_colored, scales)
    p.append(process.ResizingPicture(parameter))
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def rotate(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  is_colored: bool = False,
  degree: float = 0.0,
):
  """api to rotate movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      degree (float, optional): degree of rotation. Defaults to 0.0.
  """
  if not movie and not picture:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie:
    parameter = process.RotatingParameter(movie, is_colored, degree)
    p.append(process.RotatingMovie(parameter))
  if picture:
    parameter = process.RotatingParameter(picture, is_colored, degree)
    p.append(process.RotatingPicture(parameter))
  image_process = process.ProcessesExecution(p)
  image_process.execute()
