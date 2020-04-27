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
  target_list: Optional[List[str]] = None,
  is_colored: bool = False,
  fps: Optional[float] = None,
):
  """api to animate pictures (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      fps (float, optional): fps of created movie. Defaults to None. if this is not
      given, you will select this in GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no picture is given!")
    return (None, [])

  return process.AnimatingPicture(
    picture_list=target_list, is_colored=is_colored, fps=fps
  ).execute()


def binarize(
  *,
  target_list: Optional[List[str]] = None,
  is_movie: bool = False,
  thresholds: Tuple[int, int] = None,
):
  """api to binarize movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movies, pictures or directories where pictures are stored. Defaults to None.
      is_movie (bool, optional): movie (True) or picture (False). Defaults to False.
      thresholds (Tuple[int, int], optional): [low, high] threshold values to be used to binarize movie/picture. Low threshold must be smaller than high one. If this variable is None, this will be selected using GUI window. Defaults to None.

   Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no target is given!")
    return (None, [])

  if is_movie:
    return process.BinarizingMovie(
      movie_list=target_list, thresholds=thresholds
    ).execute()

  else:
    return process.BinarizingPicture(
      picture_list=target_list, thresholds=thresholds
    ).execute()


def capture(
  *,
  target_list: List[str] = None,
  is_colored: bool = False,
  times: Tuple[float, float, float] = None,
):
  """api to capture movies (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movie-file paths. Defaults to None.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). Start must be smaller than stop, and difference between start and stop must be larger than step. Defaults to None. If this variable is None, this will be selected using GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no movie is given!")
    return (None, [])

  return process.CapturingMovie(
    movie_list=target_list, is_colored=is_colored, times=times
  ).execute()


def concatenate(
  *,
  target_list: Optional[List[str]] = None,
  is_movie: bool = False,
  is_colored: bool = False,
  number: Optional[int] = None,
):
  """api to concatenate movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movies, pictures or directories where pictures are stored. Defaults to None.
      is_movie (bool, optional): movie (True) or picture (False). Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      number (int, optional): number of targets concatenated in x direction. max number of targets in each direction is 25. if this variable is None, this will be selected using GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no target is given!")
    return (None, [])

  if is_movie:
    return process.ConcatenatingMovie(
      movie_list=target_list, is_colored=is_colored, number=number
    ).execute()

  else:
    return process.ConcatenatingPicture(
      picture_list=target_list, is_colored=is_colored, number=number
    ).execute()


def crop(
  *,
  target_list: Optional[List[str]] = None,
  is_movie: bool = False,
  is_colored: bool = False,
  positions: Optional[Tuple[int, int, int, int]] = None,
):
  """api to crop movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movies, pictures or directories where pictures are stored. Defaults to None.
      is_movie (bool, optional): movie (True) or picture (False). Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      postisions (Tuple[int, int, int, int], optional): [x_1, y_1,x_2, y_2] two positions to crop movie/picture. position_1 must be smaller than position_2 Defaults to None. If this variable is None, this will be selected using GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no target is given!")
    return (None, [])

  if is_movie:
    return process.CroppingMovie(
      movie_list=target_list, is_colored=is_colored, positions=positions
    ).execute()

  else:
    return process.CroppingPicture(
      picture_list=target_list, is_colored=is_colored, positions=positions
    ).execute()


def hist_luminance(
  *, target_list: Optional[List[str]] = None, is_colored: bool = False
):
  """api to create histgram of luminance (bgr or gray) of picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of paths of pictures or directories where pictures are stored. Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no picture is given!")
    return (None, [])

  return process.CreatingLuminanceHistgramPicture(
    picture_list=target_list, is_colored=is_colored
  ).execute()


def resize(
  *,
  target_list: Optional[List[str]] = None,
  is_movie: bool = False,
  is_colored: bool = False,
  scales: Tuple[float, float] = None,
):
  """api to resize movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movies, pictures or directories where pictures are stored. Defaults to None.
      is_movie (bool, optional): movie (True) or picture (False). Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      scales (Tuple[float, float], optional): [x, y] ratios in each direction to scale movie/picture. Defaults to None. if this is not given, you will select this in GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no target is given!")
    return (None, [])

  if is_movie:
    return process.ResizingMovie(
      movie_list=target_list, is_colored=is_colored, scales=scales
    ).execute()

  else:
    return process.ResizingPicture(
      picture_list=target_list, is_colored=is_colored, scales=scales
    ).execute()


def rotate(
  *,
  target_list: Optional[List[str]] = None,
  is_movie: bool = False,
  is_colored: bool = False,
  degree: Optional[float] = None,
):
  """api to rotate movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str], optional): list of movies, pictures or directories where pictures are stored. Defaults to None.
      is_movie (bool, optional): movie (True) or picture (False). Defaults to False.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      degree (float, optional): degree of rotation. Defaults to None. if this is not given, you will select this in GUI window

  Returns:
      return (Tuple[str, List[str]]): return type (None, "picture", "movie") and list of outputs (movies, pictures or directories where pictures are stored). Defaults to None. if process is not executed, (None, []) is returned
  """
  if not target_list:
    print("no target is given!")
    return (None, [])

  if is_movie:
    return process.RotatingMovie(
      movie_list=target_list, is_colored=is_colored, degree=degree
    ).execute()

  else:
    return process.RotatingPicture(
      picture_list=target_list, is_colored=is_colored, degree=degree
    ).execute()
