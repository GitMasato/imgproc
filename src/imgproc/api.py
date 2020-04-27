"""api for image process functions.

Basic required arguments are list of movies or pictures (or directories where pictures are stored). if no movie or picture is given, function will not be processed (print error message and return False)

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

see usage of each api function below;

"""
from typing import List, Tuple, Optional
import sys
from imgproc import process


def animate(
  target_list: List[str], *, is_colored: bool = False, fps: Optional[float] = None,
):
  """api to animate pictures (note: keyword-only argument)

  Args:
      target_list (List[str]): list of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      fps (float, optional): fps of created movie. Defaults to None. if this is not
      given, you will select this in GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not p_list and not d_list:
    sys.exit("no picture, directory is given!")

  if p_list:
    r = process.AnimatingPicture(
      target_list=p_list, is_colored=is_colored, fps=fps
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.AnimatingPictureDirectory(
      target_list=d_list, is_colored=is_colored, fps=fps
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def binarize(
  target_list: List[str], *, thresholds: Tuple[int, int] = None,
):
  """api to binarize movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movies, pictures or directories where pictures are stored.
      thresholds (Tuple[int, int], optional): [low, high] threshold values to be used to binarize movie/picture. Low threshold must be smaller than high one. If this variable is None, this will be selected using GUI window. Defaults to None.

   Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    r = process.BinarizingMovie(target_list=m_list, thresholds=thresholds).execute()
    if r is not None:
      return_list.extend(r)

  if p_list:
    r = process.BinarizingPicture(target_list=p_list, thresholds=thresholds).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.BinarizingPictureDirectory(
      target_list=d_list, thresholds=thresholds
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def capture(
  target_list: List[str],
  *,
  is_colored: bool = False,
  times: Tuple[float, float, float] = None,
):
  """api to capture movies (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movie-file paths.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). Start must be smaller than stop, and difference between start and stop must be larger than step. Defaults to None. If this variable is None, this will be selected using GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list:
    sys.exit("no movie is given!")

  if m_list:
    r = process.CapturingMovie(
      target_list=m_list, is_colored=is_colored, times=times
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def concatenate(
  target_list: List[str], *, is_colored: bool = False, number: Optional[int] = None,
):
  """api to concatenate movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movies, pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      number (int, optional): number of targets concatenated in x direction. max number of targets in each direction is 25. if this variable is None, this will be selected using GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    r = process.ConcatenatingMovie(
      target_list=m_list, is_colored=is_colored, number=number
    ).execute()
    if r is not None:
      return_list.extend(r)

  if p_list:
    r = process.ConcatenatingPicture(
      target_list=p_list, is_colored=is_colored, number=number
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.ConcatenatingPictureDirectory(
      target_list=d_list, is_colored=is_colored, number=number
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def crop(
  target_list: List[str],
  *,
  is_colored: bool = False,
  positions: Optional[Tuple[int, int, int, int]] = None,
):
  """api to crop movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movies, pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      postisions (Tuple[int, int, int, int], optional): [x_1, y_1,x_2, y_2] two positions to crop movie/picture. position_1 must be smaller than position_2 Defaults to None. If this variable is None, this will be selected using GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    r = process.CroppingMovie(
      target_list=m_list, is_colored=is_colored, positions=positions
    ).execute()
    if r is not None:
      return_list.extend(r)

  if p_list:
    r = process.CroppingPicture(
      target_list=p_list, is_colored=is_colored, positions=positions
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.CroppingPictureDirectory(
      target_list=d_list, is_colored=is_colored, positions=positions
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def hist_luminance(target_list: List[str], *, is_colored: bool = False):
  """api to create histgram of luminance (bgr or gray) of picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of paths of pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not p_list and not d_list:
    sys.exit("no picture, directory is given!")

  if p_list:
    r = process.CreatingLuminanceHistgramPicture(
      target_list=p_list, is_colored=is_colored
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.CreatingLuminanceHistgramPictureDirectory(
      target_list=d_list, is_colored=is_colored
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def resize(
  target_list: List[str],
  *,
  is_colored: bool = False,
  scales: Tuple[float, float] = None,
):
  """api to resize movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movies, pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      scales (Tuple[float, float], optional): [x, y] ratios in each direction to scale movie/picture. Defaults to None. if this is not given, you will select this in GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    r = process.ResizingMovie(
      target_list=m_list, is_colored=is_colored, scales=scales
    ).execute()
    if r is not None:
      return_list.extend(r)

  if p_list:
    r = process.ResizingPicture(
      target_list=p_list, is_colored=is_colored, scales=scales
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.ResizingPictureDirectory(
      target_list=d_list, is_colored=is_colored, scales=scales
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None


def rotate(
  target_list: List[str], *, is_colored: bool = False, degree: Optional[float] = None,
):
  """api to rotate movie/picture (note: keyword-only argument)

  Args:
      target_list (List[str]): list of movies, pictures or directories where pictures are stored.
      is_colored (bool, optional): flag to output in color. Defaults to False.
      degree (float, optional): degree of rotation. Defaults to None. if this is not given, you will select this in GUI window

  Returns:
      return (List[str], optional): list of processed pictures, directories where      pictures are stored, and movies. if no process is executed, None is returned
  """
  return_list: List[str] = []
  if not target_list:
    sys.exit("no target is given!")
  m_list, p_list, d_list = process.sort_target_type(target_list)
  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    r = process.RotatingMovie(
      target_list=m_list, is_colored=is_colored, degree=degree
    ).execute()
    if r is not None:
      return_list.extend(r)

  if p_list:
    r = process.RotatingPicture(
      target_list=p_list, is_colored=is_colored, degree=degree
    ).execute()
    if r is not None:
      return_list.extend(r)

  if d_list:
    r = process.RotatingPictureDirectory(
      target_list=d_list, is_colored=is_colored, degree=degree
    ).execute()
    if r is not None:
      return_list.extend(r)

  return return_list if return_list else None
