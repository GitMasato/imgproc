"""api for image process functions.

Basic required arguments are list of movies or pictures (or directories where pictures are stored). if no movie or picture is given, function will not be processed (print error message and return False)

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

see usage of each api function below;

"""
from typing import List, Tuple, Optional
from imgproc import process


def animate(
  *,
  picture_list: Optional[List[str]] = None,
  is_colored: bool = False,
  fps: Optional[float] = 20.0,
):
  """api to animate pictures (note: keyword-only argument)

  Args:
      picture (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      fps (float, optional): fps of created movie. Defaults to 20.0.
  """
  if not picture_list:
    print("no picture is given!")
    return False

  image_process = process.ProcessExecution(
    process.AnimatingPicture(picture_list=picture_list, is_colored=is_colored, fps=fps)
  )
  image_process.execute()


def binarize(
  *,
  movie_list: Optional[List[str]] = None,
  picture_list: Optional[List[str]] = None,
  thresholds: Tuple[int, int] = None,
):
  """api to binarize movie/picture (note: keyword-only argument)

  Args:
      movie_list (List[str], optional): list of movie-file paths. Defaults to None.
      picture_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to None.
      thresholds (Tuple[int, int], optional): [low, high] threshold values to be used to binarize movie/picture. Low threshold must be smaller than high one. If this variable is None, this will be selected using GUI window. Defaults to None.
  """
  if not movie_list and not picture_list:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie_list:
    p.append(process.BinarizingMovie(movie_list=movie_list, thresholds=thresholds))
  if picture_list:
    p.append(
      process.BinarizingPicture(picture_list=picture_list, thresholds=thresholds)
    )
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def capture(
  *,
  movie_list: List[str] = None,
  is_colored: bool = False,
  times: Tuple[float, float, float] = None,
):
  """api to capture movies (note: keyword-only argument)

  Args:
      movie_list (List[str], optional): list of movie-file paths. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). Start must be smaller than stop, and difference between start and stop must be larger than step. Defaults to None. If this variable is None, this will be selected using GUI window
  """
  if not movie_list:
    print("no movie is given!")
    return False

  image_process = process.ProcessExecution(
    process.CapturingMovie(movie_list=movie_list, is_colored=is_colored, times=times)
  )
  image_process.execute()


def crop(
  *,
  movie_list: Optional[List[str]] = None,
  picture_list: Optional[List[str]] = None,
  is_colored: bool = False,
  positions: Optional[Tuple[int, int, int, int]] = None,
):
  """api to crop movie/picture (note: keyword-only argument)

  Args:
      movie_list (List[str], optional): list of movie-file paths. Defaults to None.
      picture_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      postisions (Tuple[int, int, int, int], optional): [x_1, y_1,x_2, y_2] two positions to crop movie/picture. position_1 must be smaller than position_2 Defaults to None. If this variable is None, this will be selected using GUI window
  """
  if not movie_list and not picture_list:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie_list:
    p.append(
      process.CroppingMovie(
        movie_list=movie_list, is_colored=is_colored, positions=positions
      )
    )
  if picture_list:
    p.append(
      process.CroppingPicture(
        picture_list=picture_list, is_colored=is_colored, positions=positions
      )
    )
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def hist_luminance(
  *, picture_list: Optional[List[str]] = None, is_colored: bool = False
):
  """api to create histgram of luminance (bgr or gray) of picture (note: keyword-only argument)

  Args:
      picture_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
  """
  if not picture_list:
    print("no picture is given!")
    return False

  image_process = process.ProcessExecution(
    process.CreatingLuminanceHistgramPicture(
      picture_list=picture_list, is_colored=is_colored
    )
  )
  image_process.execute()


def resize(
  *,
  movie_list: Optional[List[str]] = None,
  picture_list: Optional[List[str]] = None,
  is_colored: bool = False,
  scales: Tuple[float, float] = (1.0, 1.0),
):
  """api to resize movie/picture (note: keyword-only argument)

  Args:
      movie_list (List[str], optional): list of movie-file paths. Defaults to None.
      picture_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      scales (Tuple[float, float], optional): [x, y] ratios in each direction to scale movie/picture. Defaults to (1.0,1.0).
  """
  if not movie_list and not picture_list:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie_list:
    p.append(
      process.ResizingMovie(movie_list=movie_list, is_colored=is_colored, scales=scales)
    )
  if picture_list:
    p.append(
      process.ResizingPicture(
        picture_list=picture_list, is_colored=is_colored, scales=scales
      )
    )
  image_process = process.ProcessesExecution(p)
  image_process.execute()


def rotate(
  *,
  movie_list: Optional[List[str]] = None,
  picture_list: Optional[List[str]] = None,
  is_colored: bool = False,
  degree: Optional[float] = 0.0,
):
  """api to rotate movie/picture (note: keyword-only argument)

  Args:
      movie_list (List[str], optional): list of movie-file paths. Defaults to None.
      picture_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      degree (float, optional): degree of rotation. Defaults to 0.0.
  """
  if not movie_list and not picture_list:
    print("no movie and picture is given!")
    return False

  p: List[process.ABCProcess] = []
  if movie_list:
    p.append(
      process.RotatingMovie(movie_list=movie_list, is_colored=is_colored, degree=degree)
    )
  if picture_list:
    p.append(
      process.RotatingPicture(
        picture_list=picture_list, is_colored=is_colored, degree=degree
      )
    )
  image_process = process.ProcessesExecution(p)
  image_process.execute()
