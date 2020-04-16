"""api for image process functions.

Basic required arguments are list of movies or pictures. if no movie or picture is given, error will be raised

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

see usage of each api function below;

"""
from typing import List, Tuple
from image_process import exception
from image_process import process


def animate(*, picture: List[str] = None, is_colored: bool = False, fps: float = 20.0):
  """api to animate pictures (note: keyword-only argument)

  Args:
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      fps (float, optional): fps of created movie. Defaults to 20.0.

  Raises:
      exception.ArgMissingError: when no picture is given
  """
  if not picture:
    raise exception.ArgMissingError("no picture is given!")

  process_list: List[process.ABCProcess] = []
  if picture:
    process_list.append(process.AnimatingPicture(picture, is_colored, fps))
  process.process(process_list)


def binarize(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  threshold: Tuple[int, int] = None,
):
  """api to binarize movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      threshold (Tuple[int, int], optional): [low, high] threshold values to be used to binarize movie/picture. Low threshold must be smaller than high one. Defaults to None. If this variable is None, this will be selected using GUI window

  Raises:
      exception.ArgMissingError: when no picture adn movie is given
  """
  if not movie and not picture:
    raise exception.ArgMissingError("no movie and picture is given!")

  process_list: List[process.ABCProcess] = []
  if movie:
    process_list.append(process.BinarizingPicture(movie, threshold))
  if picture:
    process_list.append(process.BinarizingPicture(picture, threshold))
  process.process(process_list)


def capture(
  *,
  movie: List[str] = None,
  is_colored: bool = False,
  time: Tuple[float, float, float] = None,
):
  """api to capture movies (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      time (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). Start must be smaller than stop, and difference between start and stop must be larger than step. Defaults to None. If this variable is None, this will be selected using GUI window

  Raises:
      exception.ArgMissingError: when no movie is given
  """
  if not movie:
    raise exception.ArgMissingError("no movie is given!")

  process_list: List[process.ABCProcess] = []
  if movie:
    process_list.append(process.CapturingMovie(movie, is_colored, time))
  process.process(process_list)


def crop(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  is_colored: bool = False,
  postision: Tuple[int, int, int, int] = None,
):
  """api to crop movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      postision (Tuple[int, int, int, int], optional): [x_1, y_1,x_2, y_2] two positions to crop movie/picture. position_1 must be smaller than position_2 Defaults to None. If this variable is None, this will be selected using GUI window

  Raises:
      exception.ArgMissingError: when no picture and movie is given
  """
  if not movie and not picture:
    raise exception.ArgMissingError("no movie and picture is given!")

  process_list: List[process.ABCProcess] = []
  if movie:
    process_list.append(process.CroppingMovie(movie, is_colored, postision))
  if picture:
    process_list.append(process.CroppingPicture(picture, is_colored, postision))
  process.process(process_list)


def hist_luminance(
  *, picture: List[str] = None, is_colored: bool = False,
):
  """api to create histgram of luminance (bgr or gray) of picture (note: keyword-only argument)

  Args:
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.

  Raises:
      exception.ArgMissingError: when no picture is given
  """
  if not picture:
    raise exception.ArgMissingError("no picture is given!")

  process_list: List[process.ABCProcess] = []
  if picture:
    process_list.append(process.CreatingLuminanceHistgramPicture(picture, is_colored))
  process.process(process_list)


def resize(
  *,
  movie: List[str] = None,
  picture: List[str] = None,
  is_colored: bool = False,
  scale: Tuple[float, float] = (1.0, 1.0),
):
  """api to resize movie/picture (note: keyword-only argument)

  Args:
      movie (List[str], optional): list of movie-file paths. Defaults to None.
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      scale (Tuple[float, float], optional): [x, y] ratios in each direction to scale movie/picture. Defaults to (1.0,1.0).

  Raises:
      exception.ArgMissingError: when no picture and movie is given
  """
  if not movie and not picture:
    raise exception.ArgMissingError("no movie and picture is given!")

  process_list: List[process.ABCProcess] = []
  if movie:
    process_list.append(process.ResizingMovie(movie, is_colored, scale))
  if picture:
    process_list.append(process.ResizingPicture(picture, is_colored, scale))
  process.process(process_list)


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

  Raises:
      exception.ArgMissingError: when no picture and movie is given
  """
  if not movie and not picture:
    raise exception.ArgMissingError("no movie and picture is given!")

  process_list: List[process.ABCProcess] = []
  if movie:
    process_list.append(process.RotatingMovie(movie, is_colored, degree))
  if picture:
    process_list.append(process.RotatingPicture(picture, is_colored, degree))
  process.process(process_list)
