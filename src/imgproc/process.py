"""process module containing image process functions
"""
import cv2
import imghdr
import math
import numpy
import pathlib
from matplotlib import pyplot
from typing import List, Optional, Protocol, Tuple


def sort_target_type(target_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
  """sort input by type

  Args:
      target_list (List[str]): list of pictures or movies or directories where pictures are stored

  Returns:
      Tuple[List[str], List[str], List[str]]: list of movies, pictures, and directories
  """
  movie_list: List[str] = []
  picture_list: List[str] = []
  directory_list: List[str] = []

  for target in target_list:
    target_path = pathlib.Path(target)

    if target_path.is_dir():
      directory_list.append(target)
    elif target_path.is_file():
      image_type = imghdr.what(target)
      if image_type is not None:
        picture_list.append(target)
      else:
        movie_list.append(target)

  return (movie_list, picture_list, directory_list)


def get_movie_info(cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
  """get movie information

  Args:
      cap (cv2.VideoCapture): cv2 video object

  Returns:
      Tuple[int, int, int, float]: W, H, total frame, fps of movie
  """
  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  temp_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  temp_frames = temp_frames - int(fps) if int(fps) <= temp_frames else temp_frames
  cap.set(cv2.CAP_PROP_POS_FRAMES, temp_frames)

  # CAP_PROP_FRAME_COUNT is usually not correct.
  # so take temporal frame number first, then find the correct number
  while True:
    ret, frame = cap.read()
    if not ret:
      break

  frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

  return (W, H, frames, fps)


def get_output_path(target_list: List[str], output: str) -> List[pathlib.Path]:
  """get output path list

  Args:
      target_list (List[str]): list of pictures or movies or directories where pictures are stored
      output (str): name of the lowest level directory (without path)

  Returns:
      List[pathlib.Path]: list of pathlib.Path object
  """
  output_path_list: List[pathlib.Path] = []

  for target in target_list:
    target_path = pathlib.Path(target).resolve()

    if not target_path.is_dir() and not target_path.is_file():
      print("'{0}' does not exist!".format(str(target_path)))
      continue

    layers = target_path.parts

    if "cv2" in layers:
      p_path = target_path.parents[1] if target_path.is_file() else target_path.parent
      output_path_list.append(pathlib.Path(p_path / output))
    else:
      p = target_path.stem if target_path.is_file() else target_path.name
      output_path_list.append(pathlib.Path(pathlib.Path.cwd() / "cv2" / p / output))

  for output_path in output_path_list:
    output_path.mkdir(parents=True, exist_ok=True)

  return output_path_list


def get_rotated_size(W: int, H: int, degree: float) -> Tuple[int, int]:
  """get size of rotated frame

  Args:
      W (int): width
      H (int): height
      degree (float): degree of rotation

  Returns:
      Tuple[int, int]: [size of W rotated, size of H rotated]
  """
  rad = degree / 180.0 * numpy.pi
  sin_rad = numpy.absolute(numpy.sin(rad))
  cos_rad = numpy.absolute(numpy.cos(rad))
  W_rot = int(numpy.round(H * sin_rad + W * cos_rad))
  H_rot = int(numpy.round(H * cos_rad + W * sin_rad))
  return (W_rot, H_rot)


def get_rotate_affine_matrix(
  center: Tuple[int, int], center_rot: Tuple[int, int], degree: float
) -> numpy.array:
  """get affine_matrix to rotate picture

  Args:
      center (Tuple[int,int]): center position of picture before rotation [W, H]
      center_rot (Tuple[int,int]): center position of picture after rotation [W, H]
      degree (float): degree of rotation

  Returns:
      numpy.array: affine_matrix
  """
  rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
  affine_matrix = rotation_matrix.copy()
  affine_matrix[0][2] = affine_matrix[0][2] - center[0] + center_rot[0]
  affine_matrix[1][2] = affine_matrix[1][2] - center[1] + center_rot[1]
  return affine_matrix


def get_rotated_frame(img: numpy.array, degree: float) -> numpy.array:
  """get rotated cv2 object

  Args:
      img (numpy.array): cv2 object
      degree (float): degree of rotation

  Returns:
      numpy.array: rotated cv2 object
  """
  W, H = img.shape[1], img.shape[0]
  size_rot = get_rotated_size(W, H, degree)
  center_rot = (int(size_rot[0] / 2), int(size_rot[1] / 2))
  affine = get_rotate_affine_matrix((int(W / 2), int(H / 2)), center_rot, degree)
  return cv2.warpAffine(img, affine, size_rot, flags=cv2.INTER_CUBIC)


def no(no):
  """call back function and meaningless"""
  pass


def mouse_on_select_positions(event, x, y, flags, params):
  """call back function on mouse click
  """
  points = params

  if event == cv2.EVENT_LBUTTONUP:
    points.append([x, y])


def press_q_or_Esc(key_input: int) -> bool:
  """call back function when to press q or Esq"""

  if key_input == ord("q"):
    cv2.destroyAllWindows()
    print("'q' is pressed. abort")
    return True

  elif key_input == 27:
    cv2.destroyAllWindows()
    print("'Esc' is pressed. abort")
    return True

  else:
    return False


def create_frame_trackbars(cv2_window: str, input_frames: int):
  """add 'frame\n' and 'frame s\n' trackbars for cv2 GUI"""
  tick = 100 if 100 < input_frames else input_frames
  cv2.createTrackbar("frame\n", cv2_window, 0, tick - 1, no)
  cv2.createTrackbar("frame s\n", cv2_window, 0, (int)(input_frames / 100) + 1, no)


def get_value_from_frame_trackbars(cv2_window: str, input_frames: int) -> int:
  """get values from 'frame\n' and 'frame s\n' trackbars"""
  division_number = (int)(input_frames / 100) + 1
  frame = cv2.getTrackbarPos("frame\n", cv2_window) * division_number
  frame_s = cv2.getTrackbarPos("frame s\n", cv2_window)
  tgt_frame = frame + frame_s if frame + frame_s < input_frames else input_frames
  return tgt_frame


def create_scale_trackbars(cv2_window: str):
  """add 'scale x\n*0.1', 'scale x\n*1.0', 'scale y\n*0.1', 'scale y\n*1.0' trackbars for cv2 GUI"""
  cv2.createTrackbar("scale x\n*0.1", cv2_window, 10, 10, no)
  cv2.createTrackbar("scale x\n*1.0", cv2_window, 1, 10, no)
  cv2.createTrackbar("scale y\n*0.1", cv2_window, 10, 10, no)
  cv2.createTrackbar("scale y\n*1.0", cv2_window, 1, 10, no)


def get_values_from_scale_trackbars(cv2_window: str) -> Tuple[float, float]:
  """get values from 'scale x\n*0.1', 'scale x\n*1.0', 'scale y\n*0.1', 'scale y\n*1.0' trackbars"""
  s_x_01 = cv2.getTrackbarPos("scale x\n*0.1", cv2_window)
  s_x_10 = cv2.getTrackbarPos("scale x\n*1.0", cv2_window)
  s_y_01 = cv2.getTrackbarPos("scale y\n*0.1", cv2_window)
  s_y_10 = cv2.getTrackbarPos("scale y\n*1.0", cv2_window)
  s_x = (1 if s_x_01 == 0 else s_x_01) * 0.1 * (1 if s_x_10 == 0 else s_x_10)
  s_y = (1 if s_y_01 == 0 else s_y_01) * 0.1 * (1 if s_y_10 == 0 else s_y_10)
  return s_x, s_y


def create_degree_trackbars(cv2_window: str):
  """add 'degree\n' and 'degree s\n' trackbars for cv2 GUI"""
  cv2.createTrackbar("degree\n", cv2_window, 0, 4, no)
  cv2.createTrackbar("degree s\n", cv2_window, 0, 90, no)


def get_value_from_degree_trackbars(cv2_window: str) -> int:
  """get value from 'degree\n' and 'degree s\n' trackbars"""
  degree_l = cv2.getTrackbarPos("degree\n", cv2_window) * 90
  degree_s = cv2.getTrackbarPos("degree s\n", cv2_window)
  return degree_l + degree_s


def get_values_from_bool_trackbar(
  cv2_window: str, trackbar_name: str, is_trackbar_on: bool
) -> Tuple[bool, bool]:
  """get bool values ([bool status, if status is changed]) from bool trackbar"""
  if not is_trackbar_on:
    if cv2.getTrackbarPos(trackbar_name, cv2_window):
      return True, True
    else:
      return False, False
  else:
    if not cv2.getTrackbarPos(trackbar_name, cv2_window):
      return False, True
    else:
      return True, False


def create_start_bool_trackbar(cv2_window: str):
  """add 'start cap\n' trackbar for cv2 GUI"""
  cv2.createTrackbar("start cap\n", cv2_window, 0, 1, no)


def get_values_from_start_trackbar(
  cv2_window: str, is_start_on: bool
) -> Tuple[bool, bool]:
  """get values from 'start cap\n' trackbar"""
  return get_values_from_bool_trackbar(cv2_window, "start cap\n", is_start_on)


def create_stop_bool_trackbar(cv2_window: str):
  """add 'stop cap\n' trackbar for cv2 GUI"""
  cv2.createTrackbar("stop cap\n", cv2_window, 0, 1, no)


def get_values_from_stop_trackbar(
  cv2_window: str, is_stop_on: bool
) -> Tuple[bool, bool]:
  """get values from 'stop cap\n' trackbar"""
  return get_values_from_bool_trackbar(cv2_window, "stop cap\n", is_stop_on)


def add_texts(image: numpy.array, texts: List[str], position: Tuple[int, int]):
  """add texts into cv2 object

  Args:
      image (numpy.array): cv2 image object
      texts (List[str]): texts
      position (Tuple[int, int]): position to add texts
  """
  for id, text in enumerate(texts):
    pos = (position[0], position[1] + 30 * (id + 1))
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 10)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)


def add_texts_upper_left(image: numpy.array, texts: List[str]):
  """add texts into upper left corner of cv2 object

  Args:
      image (numpy.array): cv2 image object
      texts (List[str]): texts
  """
  pos = (5, 0)
  add_texts(image, texts, pos)


def add_texts_upper_right(image: numpy.array, texts: List[str]):
  """add texts into upper right corner of cv2 object

  Args:
      image (numpy.array): cv2 image object
      texts (List[str]): texts
  """
  W = image.shape[1]
  str_len = max([len(text) for text in texts])
  pos = (W - (18 * str_len), 0)
  add_texts(image, texts, pos)


def add_texts_lower_left(image: numpy.array, texts: List[str]):
  """add texts into lower left corner of cv2 object

  Args:
      image (numpy.array): cv2 image object
      texts (List[str]): texts
  """
  H = image.shape[0]
  pos = (5, H - 15 - (30 * len(texts)))
  add_texts(image, texts, pos)


def add_texts_lower_right(image: numpy.array, texts: List[str]):
  """add texts into lower right corner of cv2 object

  Args:
      image (numpy.array): cv2 image object
      texts (List[str]): texts
  """
  W, H = image.shape[1], image.shape[0]
  str_len = max([len(text) for text in texts])
  pos = (W - (18 * str_len), H - 15 - (30 * len(texts)))
  add_texts(image, texts, pos)


class ABCProcess(Protocol):
  """abstract base class for image processing class"""

  def execute(self):
    pass


class AnimatingPicture:
  """class to create animation (movie) from pictures"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__fps = fps

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_fps(self) -> Optional[float]:
    return self.__fps

  def execute(self):
    """animating process to create movie

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    pictures = self.__get_target_list()
    output_path_list = get_output_path(pictures, "animated")
    W_list: List[int] = []
    H_list: List[int] = []
    name = (
      str(pathlib.Path(output_path_list[0] / pathlib.Path(pictures[0]).stem)) + ".mp4"
    )

    if self.__get_fps() is not None:
      fps = self.__get_fps()
    else:
      fps = select_fps(name, pictures)

    if fps is None:
      print("animate: abort!")
      return None

    for picture in pictures:
      img = cv2.imread(picture)
      W_list.append(img.shape[1])
      H_list.append(img.shape[0])

    W = max(W_list)
    H = max(H_list)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(name, fourcc, fps, (W, H), self.__is_colored())
    print("animating pictures...")

    for picture in pictures:

      img = cv2.imread(picture)
      img_show = numpy.zeros((H, W, 3), numpy.uint8)
      img_show[: img.shape[0], : img.shape[1]] = img[:]
      output.write(
        img_show if self.__is_colored() else cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
      )

    return [name]


class AnimatingPictureDirectory:
  """class to create animation (movie) from pictures"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__fps = fps

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_fps(self) -> Optional[float]:
    return self.__fps

  def execute(self):
    """animating process to create movie

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    directories = self.__get_target_list()
    output_path_list = get_output_path(directories, "animated")
    return_list: List[str] = []

    for directory, output_path in zip(directories, output_path_list):

      directory_path = pathlib.Path(directory)
      output_name = str(pathlib.Path(output_path / directory_path.name)) + ".mp4"
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self.__get_fps() is not None:
        fps = self.__get_fps()
      else:
        fps = select_fps(output_name, p_list)

      if fps is None:
        continue

      W_list: List[int] = []
      H_list: List[int] = []

      for picture in p_list:
        img = cv2.imread(picture)
        W_list.append(img.shape[1])
        H_list.append(img.shape[0])

      W = max(W_list)
      H = max(H_list)
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (W, H), self.__is_colored())
      print("animating pictures... in '{0}'".format(directory))

      for picture in p_list:

        img = cv2.imread(picture)
        img_show = numpy.zeros((H, W, 3), numpy.uint8)
        img_show[0 : img.shape[0], 0 : img.shape[1]] = img[:]
        output.write(
          img_show
          if self.__is_colored()
          else cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
        )

    return return_list if return_list else None


def select_fps(movie: str, picture_path_list: List[pathlib.Path]) -> Optional[float]:
  """select(get) fps for animating using GUI window

  Args:
      movie (str): movie name
      picture_path_list (List[pathlib.Path]): picture path list

  Returns:
      Optional[float]: fps of movie
  """
  cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(movie, 500, 700)
  help_exists = False

  frames = len(picture_path_list) - 1

  create_frame_trackbars(movie, frames)
  cv2.createTrackbar("fps\n", movie, 10, 50, no)

  print("--- animate ---")
  print("select fps in GUI window!")
  print("(s: save, h:on/off help, q/esc: abort)")

  while True:

    tgt_frame = get_value_from_frame_trackbars(movie, frames)
    fps_read = cv2.getTrackbarPos("fps\n", movie)
    fps = 1 if fps_read == 0 else fps_read
    img = cv2.imread(str(picture_path_list[tgt_frame]))

    if help_exists:
      add_texts_upper_left(
        img,
        [
          "[animate]",
          "select fps",
          "now: {0}".format(tgt_frame),
          "fps: {0}".format(fps),
        ],
      )
      add_texts_lower_right(img, ["s:save", "h:on/off help", "q/esc:abort"])

    cv2.imshow(movie, img)
    k = cv2.waitKey(1) & 0xFF

    if k == ord("s"):
      print("'s' is pressed. fps is saved ({0})".format(fps))
      cv2.destroyAllWindows()
      return float(fps)

    elif k == ord("h"):
      help_exists = False if help_exists else True

    else:
      if press_q_or_Esc(k):
        None


class BinarizingMovie:
  """class to binarize movie
  """

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of movies. Defaults to None.
        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__thresholds = thresholds

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def execute(self):
    """binarizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    if self.__get_thresholds() is not None:
      thresholds = self.__get_thresholds()
      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "binarized")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_thresholds() is None:
        thresholds = self.__select_thresholds(output_name, frames, fps, cap)
        if thresholds is None:
          continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (int(W), int(H)), False)
      print("binarizing movie '{0}'...".format(movie))

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
        ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
        output.write(bin_2)

    return return_list if return_list else None

  def __select_thresholds(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = False

    create_frame_trackbars(movie, frames)
    cv2.createTrackbar("low\n", movie, 0, 255, no)
    cv2.createTrackbar("high\n", movie, 255, 255, no)

    print("--- binarize ---")
    print("select threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)
      low = cv2.getTrackbarPos("low\n", movie)
      high = cv2.getTrackbarPos("high\n", movie)

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      ret, bin_1 = cv2.threshold(img, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(
          bin_2,
          [
            "[binarize]",
            "select thresholds",
            "now: {0:.2f}s".format(tgt_frame / fps),
            "low: {0}".format(low),
            "high: {0}".format(high),
          ],
        )
        add_texts_lower_right(bin_2, ["s:save", "h:on/off help", "q/esc:abort"])

      cv2.imshow(movie, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if high <= low:
          print("high luminance threshold must be > low")
          continue

        print("'s' is pressed. threshold is saved ({0}, {1})".format(low, high))
        cv2.destroyAllWindows()
        return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class BinarizingPicture:
  """class to binarize picture
  """

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__thresholds = thresholds

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def execute(self):
    """binarizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    if self.__get_thresholds() is not None:
      thresholds = self.__get_thresholds()
      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        return None

    pictures = self.__get_target_list()
    output_path_list = get_output_path(pictures, "binarized")
    return_list: List[str] = []

    for picture, output_path in zip(pictures, output_path_list):

      picture_path = pathlib.Path(picture)
      name = str(pathlib.Path(output_path / picture_path.name))
      img = cv2.imread(picture)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      if self.__get_thresholds() is None:
        thresholds = self.__select_thresholds(name, gray)
        if thresholds is None:
          continue

      print("binarizing picture '{0}'...".format(picture))
      ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
      return_list.append(name)
      cv2.imwrite(name, bin_2)

    return return_list if return_list else None

  def __select_thresholds(
    self, picture: str, img: numpy.array
  ) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        picture (str): name of image
        img (numpy.array): cv2 image object

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    help_exists = False

    cv2.createTrackbar("low\n", picture, 0, 255, no)
    cv2.createTrackbar("high\n", picture, 255, 255, no)
    print("--- binarize ---")
    print("select threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    while True:

      low = cv2.getTrackbarPos("low\n", picture)
      high = cv2.getTrackbarPos("high\n", picture)
      ret, bin_1 = cv2.threshold(img, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(bin_2, ["binarize:", "select threshold"])
        add_texts_lower_right(bin_2, ["s:save", "h:on/off help", "q/esc:abort"])

      cv2.imshow(picture, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if high <= low:
          print("high luminance threshold must be > low")
          continue

        print("'s' is pressed. threshold is saved ({0}, {1})".format(low, high))
        cv2.destroyAllWindows()
        return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class BinarizingPictureDirectory:
  """class to binarize picture
  """

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__thresholds = thresholds

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def execute(self):
    """binarizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    if self.__get_thresholds() is not None:
      thresholds = self.__get_thresholds()
      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        return (None, [])

    directories = self.__get_target_list()
    output_path_list = get_output_path(directories, "binarized")
    return_list: List[str] = []

    for directory, output_path in zip(directories, output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]
      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self.__get_thresholds() is None:
        thresholds = self.__select_thresholds(str(output_path), p_list)
        if thresholds is None:
          continue

      return_list.append(str(output_path))
      print("binarizing picture in '{0}'...".format(directory))

      for p in p_list:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
        ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
        cv2.imwrite(str(pathlib.Path(output_path / pathlib.Path(p).name)), bin_2)

    return return_list if return_list else None

  def __select_thresholds(
    self, directory: str, picture_list: List[str]
  ) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        directory (str): directory name
        picture_ist (List[str]): picture list

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = False

    frames = len(picture_list) - 1

    create_frame_trackbars(directory, frames)
    cv2.createTrackbar("low\n", directory, 0, 255, no)
    cv2.createTrackbar("high\n", directory, 255, 255, no)

    print("--- binarize ---")
    print("select threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    while True:

      tgt_frame = get_value_from_frame_trackbars(directory, frames)
      low = cv2.getTrackbarPos("low\n", directory)
      high = cv2.getTrackbarPos("high\n", directory)

      img = cv2.imread(picture_list[tgt_frame])
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(
          bin_2,
          [
            "[binarize]",
            "select thresholds",
            "now: {0}".format(tgt_frame),
            "low: {0}".format(low),
            "high: {0}".format(high),
          ],
        )
        add_texts_lower_right(bin_2, ["s:save", "h:on/off help", "q/esc:abort"])

      cv2.imshow(directory, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if high <= low:
          print("high luminance threshold must be > low")
          continue

        print("'s' is pressed. threshold is saved ({0}, {1})".format(low, high))
        cv2.destroyAllWindows()
        return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class CapturingMovie:
  """class to capture movie"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    times: Optional[Tuple[float, float, float]] = None,
  ):
    """constructor

    Args:
        movie_list (List[str], optional): list of movies. Defaults to None.
        is_colored (bool, optional):  flag to output in color. Defaults to False.
        times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). If this variable is None, this will be selected using GUI window Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__times = times

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_times(self) -> Optional[Tuple[float, float, float]]:
    return self.__times

  def execute(self):
    """capturing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (directory) path names
    """
    if self.__get_times() is not None:
      time = self.__get_times()

      if time[1] - time[0] <= time[2]:
        print("difference between stop and start must be > time step")
        return None
      if time[1] <= time[0]:
        print("stop must be > start")
        return None
      if time[2] < 0.001:
        print("time step must be > 0")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "captured")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_times() is None:
        time = self.__select_times(str(output_path), frames, fps, cap)

      if time is None:
        continue

      if frames < round(time[0] * fps) or frames < round(time[1] * fps):
        f1, f2 = round(time[0] * fps), round(time[1] * fps)
        print("start & stop ({0},{1}) must be < max frame ({2})".format(f1, f2, frames))
        continue

      capture_time = time[0]
      return_list.append(str(output_path))
      print("capturing movie '{0}'...".format(movie))

      while capture_time <= time[1]:

        cap.set(cv2.CAP_PROP_POS_FRAMES, round(capture_time * fps))
        ret, frame = cap.read()
        if not ret:
          break
        pic_time = int(round(capture_time - time[0], 3) * 1000)
        pic_name = "{0}/{1:08}_ms.png".format(str(output_path), pic_time)
        f = frame if self.__is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pic_name, f)
        capture_time += time[2]

    return return_list if return_list else None

  def __select_times(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[float, float, float]]:
    """select(get) parametes for capture using GUI window

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[float, float, float]]: [start, stop, step] parameters for capturing movie (s).
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists, is_start_on, is_stop_on = False, False, False
    start_time, stop_time = 0.0, 0.0

    print("--- capture ---\nselect time (start, stop, step) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(movie, frames)
    create_start_bool_trackbar(movie)
    create_stop_bool_trackbar(movie)
    cv2.createTrackbar("step 10ms\n", movie, 100, 100, no)

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)
      time_step = cv2.getTrackbarPos("step 10ms\n", movie) * 10 * 0.001

      is_start_on, is_bool_changed = get_values_from_start_trackbar(movie, is_start_on)
      if is_bool_changed:
        start_time = tgt_frame / fps if is_start_on else 0.0

      is_stop_on, is_bool_changed = get_values_from_stop_trackbar(movie, is_stop_on)
      if is_bool_changed:
        stop_time = tgt_frame / fps if is_stop_on else 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if help_exists:
        add_texts_upper_left(
          img,
          [
            "[capture]",
            "select start,stop,step",
            "s: save",
            "q/esc: abort",
            "now: {0:.2f}s".format(tgt_frame / fps),
            "start: {0:.2f}s".format(start_time),
            "stop: {0:.2f}s".format(stop_time),
            "step: {0:.2f}s".format(time_step),
          ],
        )

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed.")
        print("start,stop,step ({0},{1},{2})".format(start_time, stop_time, time_step))

        if (not is_start_on) or (not is_stop_on):
          print("start or stop is not selected!")
          continue
        if stop_time <= start_time:
          print("stop must be > start!")
          continue
        if stop_time - start_time <= time_step:
          print("difference between stop & start must be > step!")
          continue
        if time_step < 0.001:
          print("step must be > 0!")
          continue

        cv2.destroyAllWindows()
        return (start_time, stop_time, time_step)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          return None


class ConcatenatingMovie:
  """class to concatenate movie"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    number: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        number (Optional[int], optional): number of pictures concatenated in x direction. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__number = number

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_number(self) -> Optional[int]:
    return self.__number

  def execute(self):
    """resizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    movies = self.__get_target_list()
    size_list: List[Tuple[int, int]] = []
    frame_list: List[int] = []
    fps_list: List[float] = []

    if 25 < len(movies):
      print("'{0}' movies given. max is 25".format(len(movies)))
      return None

    for movie in movies:
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      size_list.append((W, H))
      frame_list.append(frames)
      fps_list.append(fps)

    if self.__get_number() is not None:
      number_x = self.__get_number()
    else:
      number_x = self.__select_number(movies, size_list, frame_list)

    if number_x is None:
      return None

    number_y = math.ceil(len(movies) / number_x)
    black_list = [numpy.zeros((s[1], s[0], 3), numpy.uint8) for s in size_list]
    concat = self.__get_concatenated_movie_frame(
      movies, frame_list, black_list, 0, number_x, number_y
    )

    size = (concat.shape[1], concat.shape[0])
    movie_first = movies[0]
    output_path_list = get_output_path([movie_first], "concatenated")
    output_name = (
      str(pathlib.Path(output_path_list[0] / pathlib.Path(movie_first).stem)) + ".mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
      output_name, fourcc, fps_list[0], size, self.__is_colored()
    )
    print("concatenating movie '{0}'...".format(output_name))

    for frame in range(max(frame_list) + 1):
      concat = self.__get_concatenated_movie_frame(
        movies, frame_list, black_list, frame, number_x, number_y
      )
      output.write(
        concat if self.__is_colored() else cv2.cvtColor(concat, cv2.COLOR_BGR2GRAY)
      )
    return [output_name]

  def __select_number(
    self,
    movie_list: List[str],
    size_list: List[Tuple[int, int]],
    frame_list: List[int],
  ) -> Optional[int]:
    """select(get) number of concatenating using GUI window

    Args:
        movie (List[str]): movie list
        size_list (List[Tuple[int, int]]): list of movie size
        frame_list (List[int]): list of movie frame size

    Returns:
        Optional[int]: number of movies concatenated in x direction
    """
    window = movie_list[0]
    movie_number = len(movie_list)
    frames = max(frame_list)
    black_list = [numpy.zeros((s[1], s[0], 3), numpy.uint8) for s in size_list]

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 500, 700)
    help_exists = False

    create_frame_trackbars(window, frames)
    cv2.createTrackbar("x\n", window, 1, 5, no)

    print("--- concatenate ---")
    print("select number of pictures in x direction in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    while True:

      tgt_frame = get_value_from_frame_trackbars(window, frames)
      number_x = cv2.getTrackbarPos("x\n", window)

      if number_x == 0:
        number_x = 1
      elif movie_number < number_x:
        number_x = movie_number

      number_y = math.ceil(movie_number / number_x)
      concat = self.__get_concatenated_movie_frame(
        movie_list, frame_list, black_list, tgt_frame, number_x, number_y
      )

      if help_exists:
        add_texts_upper_left(
          concat, ["[concatenate]", "select x", "frame: {0}".format(tgt_frame)],
        )
        add_texts_lower_right(concat, ["s:save", "h:on/off help", "q/esc:abort"])

      cv2.imshow(window, concat)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print(
          "'s' is pressed. concatenating number (x-dir) is saved ({0})".format(number_x)
        )
        cv2.destroyAllWindows()
        return number_x

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None

  def __get_concatenated_movie_frame(
    self,
    movie_list: List[str],
    frame_list: List[int],
    black_list: List[numpy.array],
    frame: int,
    number_x: int,
    number_y: int,
  ) -> numpy.array:
    """get concatenated frame of movie

    Args:
        movie_list (List[str]): list of movies
        frame_list (List[int]): list of frame numbers of movies
        black_list (List[numpy.array]): list of black images
        frame (int): target frame
        number_x (int): number of movies concatenated in x direction
        number_y (int): number of movies concatenated in x direction

    Returns:
        numpy.array: concatenated image
    """
    pic_list = []
    for idx, movie in enumerate(movie_list):
      if frame_list[idx] < frame:
        pic_list.append(black_list[idx])
      else:
        cap = cv2.VideoCapture(movie)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        pic_list.append(img)
    for a in range(number_x * number_y - len(movie_list)):
      pic_list.append(black_list[0])

    multi_list = [
      pic_list[y * number_x : y * number_x + number_x] for y in range(number_y)
    ]
    concat_W = [vconcat_W(one_list, pic_list[0].shape[0]) for one_list in multi_list]
    return vconcat_H(concat_W, concat_W[0].shape[1])


class ConcatenatingPicture:
  """class to concatenate picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    number: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        numbers (Optional[int], optional): number of pictures concatenated in x direction. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__number = number

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_number(self) -> Optional[int]:
    return self.__number

  def execute(self):
    """concatenating process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    pictures = self.__get_target_list()
    size_list = []

    if 25 < len(pictures):
      print("'{0}' pictures given. max is 25".format(len(pictures)))
      return None

    for picture in pictures:
      img = cv2.imread(picture)
      size_list.append((img.shape[1], img.shape[0]))

    if self.__get_number() is not None:
      number_x = self.__get_number()
    else:
      number_x = select_number(pictures, size_list)

    if number_x is None:
      return None

    output_path_list = get_output_path([pictures[0]], "concatenated")
    name = str(pathlib.Path(output_path_list[0] / pathlib.Path(pictures[0]).name))

    print("concatenating picture '{0}'...".format(name))
    number_y = math.ceil(len(pictures) / number_x)
    concat = get_concatenated_pictures(pictures, number_x, number_y)
    cv2.imwrite(
      name, concat if self.__is_colored() else cv2.cvtColor(concat, cv2.COLOR_BGR2GRAY),
    )
    return [name]


class ConcatenatingPictureDirectory:
  """class to concatenate picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    number: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        numbers (Optional[int], optional):  number of pictures concatenated in x direction. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__number = number

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_number(self) -> Optional[int]:
    return self.__number

  def execute(self):
    """concatenating process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    directories = self.__get_target_list()
    output_path_list = get_output_path(directories, "concatenated")
    return_list: List[str] = []

    for idx, directory in enumerate(directories):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]
      size_list = []
      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if 25 < len(p_list):
        print("'{0}' pictures given. max is 25".format(len(p_list)))
        continue

      for p in p_list:
        img = cv2.imread(p)
        size_list.append((img.shape[1], img.shape[0]))

      if self.__get_number() is not None:
        number_x = self.__get_number()
      else:
        number_x = select_number(p_list, size_list)

      if number_x is None:
        continue

      print("concatenating pictures in '{0}'...".format(directory))
      number_y = math.ceil(len(p_list) / number_x)
      concat = get_concatenated_pictures(p_list, number_x, number_y)
      file_name = directory_path.name + pathlib.Path(p_list[0]).suffix
      name = str(pathlib.Path(output_path_list[idx] / file_name))
      return_list.append(name)
      cv2.imwrite(
        name,
        concat if self.__is_colored() else cv2.cvtColor(concat, cv2.COLOR_BGR2GRAY),
      )

    return return_list if return_list else None


def select_number(
  picture_list: List[str], size_list: List[Tuple[int, int]],
) -> Optional[int]:
  """select(get) number of concatenating pictures in x direction using GUI window

  Args:
      picture_list (List[str]): picture list
      size_list (List[Tuple[int, int]]): list of picture size

  Returns:
      Optional[int]: number of pictures concatenated in x direction
  """
  window = picture_list[0]
  cv2.namedWindow(window, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window, 500, 700)
  help_exists = False
  cv2.createTrackbar("x\n", window, 1, 5, no)

  print("--- concatenate ---")
  print("select number of pictures in x direction in GUI window!")
  print("(s: save, h:on/off help, q/esc: abort)")

  while True:

    number_x = cv2.getTrackbarPos("x\n", window)
    if number_x == 0:
      number_x = 1
    elif len(picture_list) < number_x:
      number_x = len(picture_list)
    number_y = math.ceil(len(picture_list) / number_x)
    concat = get_concatenated_pictures(picture_list, number_x, number_y)

    if help_exists:
      add_texts_upper_left(
        concat, ["[concatenate]", "select x"],
      )
      add_texts_lower_right(concat, ["s:save", "h:on/off help", "q/esc:abort"])
    cv2.imshow(window, concat)
    k = cv2.waitKey(1) & 0xFF

    if k == ord("s"):
      print(
        "'s' is pressed. concatenating number (x-dir) is saved ({0})".format(number_x)
      )
      cv2.destroyAllWindows()
      return number_x

    elif k == ord("h"):
      help_exists = False if help_exists else True

    else:
      if press_q_or_Esc(k):
        None


def get_concatenated_pictures(
  picture_list: List[str], number_x: int, number_y: int,
) -> numpy.array:
  """get concatenated frame of movie

    Args:
        picture_list (List[str]): list of pictures
        number_x (int): number of movies concatenated in x direction
        number_y (int): number of movies concatenated in x direction

    Returns:
        numpy.array: concatenated image
    """
  pic_list = []
  for idx, picture in enumerate(picture_list):
    pic_list.append(cv2.imread(picture))

  for a in range(number_x * number_y - len(picture_list)):
    pic_list.append(
      numpy.zeros((pic_list[0].shape[0], pic_list[0].shape[1], 3), numpy.uint8)
    )

  multi_list = [
    pic_list[y * number_x : y * number_x + number_x] for y in range(number_y)
  ]
  concat_W = [vconcat_W(one_list, pic_list[0].shape[0]) for one_list in multi_list]
  return vconcat_H(concat_W, concat_W[0].shape[1])


def vconcat_H(image_list: List[numpy.array], W: int) -> numpy.array:
  """concatenate image in H direction

  Args:
      image_list (List[numpy.array]): list of images concatenated
      W (int): W size

  Returns:
      numpy.array: numpy.array image
  """
  resized = [
    cv2.resize(
      img, (W, int(img.shape[0] * W / img.shape[1])), interpolation=cv2.INTER_CUBIC,
    )
    for img in image_list
  ]
  return cv2.vconcat(resized)


def vconcat_W(image_list: List[numpy.array], H: int) -> numpy.array:
  """concatenate image in W direction

  Args:
      image_list (List[numpy.array]): list of images concatenated
      H (int): H size

  Returns:
      numpy.array: numpy.array image
  """
  resized = [
    cv2.resize(
      img, (int(H / img.shape[0] * img.shape[1]), H), interpolation=cv2.INTER_CUBIC,
    )
    for img in image_list
  ]
  return cv2.hconcat(resized)


class CroppingMovie:
  """class to crop movie"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__positions = positions

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def execute(self):
    """capturing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    if self.__get_positions() is not None:
      positions = self.__get_positions()
      if positions[2] <= positions[0] or positions[3] <= positions[1]:
        print("2nd position must be > 1st")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "cropped")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_positions() is None:
        positions = self.__select_positions(output_name, W, H, frames, fps, cap)
        if positions is None:
          continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      size = (int(positions[2] - positions[0]), int(positions[3] - positions[1]))
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self.__is_colored())
      print("cropping movie '{0}'...".format(movie))

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        cropped = frame[positions[1] : positions[3], positions[0] : positions[2]]
        output.write(
          cropped if self.__is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        )

    return return_list if return_list else None

  def __select_positions(
    self, movie: str, W: int, H: int, frames: int, fps: float, cap: cv2.VideoCapture,
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for capring process using GUI window

    Args:
        movie (str): movie file name
        W (int): W video length
        H (int): H video length
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = False

    create_frame_trackbars(movie, frames)

    points: List[Tuple[int, int]] = []
    cv2.setMouseCallback(movie, mouse_on_select_positions, points)
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if len(points) == 1:
        cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
        else:
          cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()

      if help_exists:
        add_texts_lower_right(
          img, ["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"],
        )
        add_texts_upper_left(
          img,
          ["[crop]", "select two positions", "now: {0:.2f}s".format(tgt_frame / fps)],
        )

        if len(points) == 1:
          add_texts_upper_right(
            img, ["selected:", "[{0},{1}]".format(points[0][0], points[0][1])]
          )
        elif len(points) == 2:
          if not (points[1][1] <= points[0][1] or points[1][0] <= points[0][0]):
            add_texts_upper_right(
              img,
              [
                "selected:",
                "[{0},{1}]".format(points[0][0], points[0][1]),
                "[{0},{1}]".format(points[1][0], points[1][1]),
              ],
            )

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if len(points) == 2:
          print("'s' is pressed. cropped positions are saved ({0})".format(points))
          cv2.destroyAllWindows()
          return (points[0][0], points[0][1], points[1][0], points[1][1])
        else:
          print("two positions for cropping are not selected yet")
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        continue

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class CroppingPicture:
  """class to capture picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__positions = positions

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def execute(self):
    """cropping process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    if self.__get_positions() is not None:
      positions = self.__get_positions()
      if positions[2] <= positions[0] or positions[3] <= positions[1]:
        print("2nd position must be larger than 1st")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "cropped")
    return_list: List[str] = []

    for picture, output_path in zip(self.__get_target_list(), output_path_list):

      name = str(pathlib.Path(output_path / pathlib.Path(picture).name))
      img = cv2.imread(picture)
      if self.__get_positions() is None:
        positions = self.__select_positions(name, img)
        if positions is None:
          continue

      print("cropping picture '{0}'...".format(picture))
      cropped = img[positions[1] : positions[3], positions[0] : positions[2]]
      return_list.append(name)
      cv2.imwrite(
        name,
        cropped if self.__is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
      )

    return return_list if return_list else None

  def __select_positions(
    self, picture: str, img: numpy.array
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for capring process using GUI window

    Args:
        picture (str): image name
        img (numpy.array): cv2 image object

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    W, H = img.shape[1], img.shape[0]
    points: List[Tuple[int, int]] = []

    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(picture, mouse_on_select_positions, points)
    help_exists = False
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      img_show = img.copy()

      if len(points) == 1:
        cv2.line(img_show, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img_show, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
        else:
          cv2.line(img_show, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img_show, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img_show, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img_show, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()

      if help_exists:
        add_texts_lower_right(
          img_show,
          ["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"],
        )
        add_texts_upper_left(
          img_show, ["[crop]", "select two positions"],
        )

        if len(points) == 1:
          add_texts_upper_right(
            img_show, ["selected:", "[{0},{1}]".format(points[0][0], points[0][1])]
          )
        elif len(points) == 2:
          if not (points[1][1] <= points[0][1] or points[1][0] <= points[0][0]):
            add_texts_upper_right(
              img_show,
              [
                "selected:",
                "[{0},{1}]".format(points[0][0], points[0][1]),
                "[{0},{1}]".format(points[1][0], points[1][1]),
              ],
            )

      cv2.imshow(picture, img_show)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if len(points) == 2:
          print("'s' is pressed. cropped positions are saved ({0})".format(points))
          cv2.destroyAllWindows()
          return (points[0][0], points[0][1], points[1][0], points[1][1])
        else:
          print("two positions for cropping are not selected yet")
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        continue

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class CroppingPictureDirectory:
  """class to capture picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__positions = positions

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def execute(self):
    """cropping process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    if self.__get_positions() is not None:
      positions = self.__get_positions()
      if positions[2] <= positions[0] or positions[3] <= positions[1]:
        print("2nd position must be larger than 1st")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "cropped")
    return_list: List[str] = []

    for directory, output_path in zip(self.__get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]
      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self.__get_positions() is None:
        positions = self.__select_positions(str(output_path), p_list)
        if positions is None:
          continue

      return_list.append(str(output_path))
      print("cropping picture in '{0}'...".format(directory))

      for p in p_list:
        img = cv2.imread(p)
        cropped = img[positions[1] : positions[3], positions[0] : positions[2]]
        cv2.imwrite(
          str(pathlib.Path(output_path / pathlib.Path(p).name)),
          cropped if self.__is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
        )

    return return_list if return_list else None

  def __select_positions(
    self, directory: str, picture_list: List[str]
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for capring process using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = False

    frames = len(picture_list) - 1
    create_frame_trackbars(directory, frames)

    points: List[Tuple[int, int]] = []
    cv2.setMouseCallback(directory, mouse_on_select_positions, points)
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      tgt_frame = get_value_from_frame_trackbars(directory, frames)
      img = cv2.imread(picture_list[tgt_frame])
      W, H = img.shape[1], img.shape[0]

      if len(points) == 1:
        cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
        else:
          cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()

      if help_exists:
        add_texts_lower_right(
          img, ["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"],
        )
        add_texts_upper_left(
          img, ["[crop]", "select two positions", "frame: {0}".format(tgt_frame)],
        )

        if len(points) == 1:
          add_texts_upper_right(
            img, ["selected:", "[{0},{1}]".format(points[0][0], points[0][1])]
          )
        elif len(points) == 2:
          if not (points[1][1] <= points[0][1] or points[1][0] <= points[0][0]):
            add_texts_upper_right(
              img,
              [
                "selected:",
                "[{0},{1}]".format(points[0][0], points[0][1]),
                "[{0},{1}]".format(points[1][0], points[1][1]),
              ],
            )

      cv2.imshow(directory, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if len(points) == 2:
          print("'s' is pressed. cropped positions are saved ({0})".format(points))
          cv2.destroyAllWindows()
          return (points[0][0], points[0][1], points[1][0], points[1][1])
        else:
          print("two positions for cropping are not selected yet")
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        continue

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class CreatingLuminanceHistgramPicture:
  """class to create luminance histgram of picture"""

  def __init__(
    self, *, target_list: Optional[List[str]] = None, is_colored: bool = False
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    self.__target_list = target_list
    self.__colored = is_colored

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def execute(self):
    """creating luminance histglam

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "histgram_luminance")
    return_list: List[str] = []

    for picture, output_path in zip(self.__get_target_list(), output_path_list):

      print("creating luminance histgram of picture '{0}'...".format(picture))
      return_list.append(str(pathlib.Path(output_path / pathlib.Path(picture).name)))

      if self.__is_colored():
        create_color_figure(picture, output_path)
      else:
        create_gray_figure(picture, output_path)

    return return_list if return_list else None


class CreatingLuminanceHistgramPictureDirectory:
  """class to create luminance histgram of picture"""

  def __init__(
    self, *, target_list: Optional[List[str]] = None, is_colored: bool = False
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    self.__target_list = target_list
    self.__colored = is_colored

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def execute(self):
    """creating luminance histglam

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "histgram_luminance")
    return_list: List[str] = []

    for directory, output_path in zip(self.__get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]
      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if p_list:
        return_list.append(str(output_path))

        print("creating luminance histgram of picture in '{0}'...".format(directory))

        if self.__is_colored():
          for p in p_list:
            create_color_figure(p, output_path)
        else:
          for p in p_list:
            create_gray_figure(p, output_path)

    return return_list if return_list else None


def create_color_figure(picture: str, output_path: pathlib.Path):
  """create output figure in color

  Args:
      picture (str): picture name
      output_path (pathlib.Path): output directory (pathlib.Path) object
  """
  img = cv2.imread(picture)
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  fig = pyplot.figure()
  subplot = fig.add_subplot(2, 2, 1)
  subplot.imshow(rgb)
  subplot.axis("off")
  subplot = fig.add_subplot(2, 2, 2)
  subplot.hist(rgb[:, :, 0].flatten(), bins=numpy.arange(256 + 1), color="r")
  subplot = fig.add_subplot(2, 2, 3)
  subplot.hist(rgb[:, :, 1].flatten(), bins=numpy.arange(256 + 1), color="g")
  subplot = fig.add_subplot(2, 2, 4)
  subplot.hist(rgb[:, :, 2].flatten(), bins=numpy.arange(256 + 1), color="b")
  fig.savefig(str(pathlib.Path(output_path / pathlib.Path(picture).name)))
  pyplot.close(fig)


def create_gray_figure(picture: str, output_path: pathlib.Path):
  """create output figure in gray

  Args:
      picture (str): picture name
      output_path (pathlib.Path): output directory (pathlib.Path) object
  """
  img = cv2.imread(picture)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  fig = pyplot.figure()
  subplot = fig.add_subplot(1, 2, 1)
  subplot.imshow(gray, cmap="gray")
  subplot.axis("off")
  subplot = fig.add_subplot(1, 2, 2)
  subplot.hist(gray.flatten(), bins=numpy.arange(256 + 1))
  fig.savefig(str(pathlib.Path(output_path / pathlib.Path(picture).name)))
  pyplot.close(fig)


class ResizingMovie:
  """class to resize movie"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale movie. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__scales = scales

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def execute(self):
    """resizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "resized")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_scales() is not None:
        scales = self.__get_scales()
      else:
        scales = self.__select_scales(output_name, W, H, frames, fps, cap)

      if scales is None:
        continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      size = (int(W * scales[0]), int(H * scales[1]))
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self.__is_colored())
      print("resizing movie '{0}'...".format(movie))

      while True:

        ret, frame = cap.read()
        if not ret:
          break
        f1 = cv2.resize(frame, dsize=size)
        f2 = f1 if self.__is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_scales(
    self, movie: str, W: int, H: int, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[float, float]]:
    """select(get) rotation degree using GUI window

    Args:
        movie (str): movie file name
        W (int): W length of movie
        H (int): H length of movie
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[float, float]]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = False

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(movie, frames)
    create_scale_trackbars(movie)

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)
      s_x, s_y = get_values_from_scale_trackbars(movie)

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        add_texts_upper_left(
          resized,
          [
            "[resize]",
            "select scales",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "now: {0:.2f}s".format(tgt_frame / fps),
            "scale: {0:.1f},{1:.1f}".format(s_x, s_y),
          ],
        )

      cv2.imshow(movie, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class ResizingPicture:
  """class to resize picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale movie. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__scales = scales

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def execute(self):
    """resizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "resized")
    return_list: List[str] = []

    for picture, output_path in zip(self.__get_target_list(), output_path_list):

      name = str(pathlib.Path(output_path / pathlib.Path(picture).name))
      img = cv2.imread(picture)
      W, H = img.shape[1], img.shape[0]

      if self.__get_scales() is not None:
        scales = self.__get_scales()
      else:
        scales = self.__select_scales(name, img, W, H)

      if scales is None:
        continue

      print("resizing picture '{0}'...".format(picture))
      return_list.append(name)
      f1 = cv2.resize(img, dsize=(int(W * scales[0]), int(H * scales[1])))
      f2 = f1 if self.__is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(name, f2)

    return return_list if return_list else None

  def __select_scales(
    self, picture: str, img: numpy.array, W: int, H: int
  ) -> Optional[Tuple[float, float]]:
    """select(get) resizing scales using GUI window

    Args:
        picture (str): name of image
        img (numpy.array): cv2 image object
        W (int): W length of picture
        H (int): H length of picture

    Returns:
        Optional[float]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)
    help_exists = False

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_scale_trackbars(picture)

    while True:

      s_x, s_y = get_values_from_scale_trackbars(picture)
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        add_texts_upper_left(
          resized,
          [
            "[resize]",
            "select scales",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "scale: {0:.1f},{1:.1f}".format(s_x, s_y),
          ],
        )

      cv2.imshow(picture, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class ResizingPictureDirectory:
  """class to resize picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale movie. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__scales = scales

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def execute(self):
    """resizing process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "resized")
    return_list: List[str] = []

    for directory, output_path in zip(self.__get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self.__get_scales() is not None:
        scales = self.__get_scales()
      else:
        scales = self.__select_scales(str(output_path), p_list)

      if scales is None:
        continue

      return_list.append(str(output_path))
      print("resizing picture in '{0}'...".format(directory))

      for p in p_list:

        img = cv2.imread(p)
        W, H = img.shape[1], img.shape[0]
        f1 = cv2.resize(img, dsize=(int(W * scales[0]), int(H * scales[1])))
        f2 = f1 if self.__is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        pic_name = str(pathlib.Path(output_path / pathlib.Path(p).name))
        cv2.imwrite(pic_name, f2)

    return return_list if return_list else None

  def __select_scales(
    self, directory: str, picture_list: List[str]
  ) -> Optional[Tuple[float, float]]:
    """select(get) resizing scales using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[float]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = False
    frames = len(picture_list) - 1

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(directory, frames)
    create_scale_trackbars(directory)

    while True:

      tgt_frame = get_value_from_frame_trackbars(directory, frames)
      s_x, s_y = get_values_from_scale_trackbars(directory)
      img = cv2.imread(picture_list[tgt_frame])
      W, H = img.shape[1], img.shape[0]
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        add_texts_upper_left(
          resized,
          [
            "[resize]",
            "select scales",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "frame: {0}".format(tgt_frame),
            "scale: {0:.1f},{1:.1f}".format(s_x, s_y),
          ],
        )

      cv2.imshow(directory, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class RotatingMovie:
  """class to rotate movie
  """

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): [description]. flag to output in color to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__degree = degree

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_degree(self) -> Optional[float]:
    return self.__degree

  def execute(self):
    """rotating process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "rotated")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_degree() is not None:
        degree = self.__get_degree()
      else:
        degree = self.__select_degree(output_name, frames, fps, cap)

      if degree is None:
        continue

      size_rot = get_rotated_size(W, H, degree)
      center_rot = (int(size_rot[0] / 2), int(size_rot[1] / 2))
      affine = get_rotate_affine_matrix((int(W / 2), int(H / 2)), center_rot, degree)

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size_rot, self.__is_colored())
      print("rotating movie '{0}'...".format(movie))

      while True:

        ret, frame = cap.read()
        if not ret:
          break
        f1 = cv2.warpAffine(frame, affine, size_rot, flags=cv2.INTER_CUBIC)
        f2 = f1 if self.__is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_degree(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = False

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(movie, frames)
    create_degree_trackbars(movie)

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)
      degree = get_value_from_degree_trackbars(movie)

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      rotated = get_rotated_frame(img, degree)

      if help_exists:
        add_texts_upper_left(
          rotated,
          [
            "[rotate]",
            "select degree",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "now: {0:.2f}s".format(tgt_frame / fps),
            "deg: {0}".format(degree),
          ],
        )

      cv2.imshow(movie, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree))
        cv2.destroyAllWindows()
        return float(degree)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class RotatingPicture:
  """class to rotate picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of pictures. Defaults to None.
        is_colored (bool, optional): [description]. flag to output in color to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__degree = degree

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_degree(self) -> Optional[float]:
    return self.__degree

  def execute(self):
    """rotating process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "rotated")
    return_list: List[str] = []

    for picture, output_path in zip(self.__get_target_list(), output_path_list):

      name = str(pathlib.Path(output_path / pathlib.Path(picture).name))
      img = cv2.imread(picture)

      if self.__get_degree() is not None:
        degree = self.__get_degree()
      else:
        degree = self.__select_degree(name, img)

      if degree is None:
        continue

      print("rotating picture '{0}'...".format(picture))
      return_list.append(name)
      rotated = get_rotated_frame(img, degree)
      f = rotated if self.__is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(name, f)

    return return_list if return_list else None

  def __select_degree(self, picture: str, img: numpy.array) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        picture (str): name of image
        img (numpy.array): cv2 image object

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)
    help_exists = False

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_degree_trackbars(picture)

    while True:

      degree = get_value_from_degree_trackbars(picture)
      rotated = get_rotated_frame(img, degree)

      if help_exists:
        add_texts_upper_left(
          rotated,
          [
            "[rotate]",
            "select degree",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "deg: {0}".format(degree),
          ],
        )

      cv2.imshow(picture, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree))
        cv2.destroyAllWindows()
        return float(degree)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class RotatingPictureDirectory:
  """class to rotate picture"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (Optional[List[str]], optional): list of directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): [description]. flag to output in color to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__degree = degree

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_degree(self) -> Optional[float]:
    return self.__degree

  def execute(self):
    """rotating process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output path names
    """
    output_path_list = get_output_path(self.__get_target_list(), "rotated")
    return_list: List[str] = []

    for directory, output_path in zip(self.__get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self.__get_degree() is not None:
        degree = self.__get_degree()
      else:
        degree = self.__select_degree(str(output_path), p_list)

      if degree is None:
        continue

      return_list.append(str(output_path))
      print("rotating picture in '{0}'...".format(directory))

      for p in p_list:

        img = cv2.imread(p)
        f1 = get_rotated_frame(img, degree)
        f2 = f1 if self.__is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        pic_name = str(pathlib.Path(output_path / pathlib.Path(p).name))
        cv2.imwrite(pic_name, f2)

    return return_list if return_list else None

  def __select_degree(self, directory: str, picture_list: List[str]) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = False
    frames = len(picture_list) - 1

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(directory, frames)
    create_degree_trackbars(directory)

    while True:

      tgt_frame = get_value_from_frame_trackbars(directory, frames)
      degree = get_value_from_degree_trackbars(directory)
      img = cv2.imread(picture_list[tgt_frame])
      rotated = get_rotated_frame(img, degree)

      if help_exists:
        add_texts_upper_left(
          rotated,
          [
            "[rotate]",
            "select degree",
            "s:save",
            "h:on/off help",
            "q/esc:abort",
            "frame: {0}".format(tgt_frame),
            "deg: {0}".format(degree),
          ],
        )

      cv2.imshow(directory, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree))
        cv2.destroyAllWindows()
        return float(degree)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None


class TrimmingMovie:
  """class to trim movie"""

  def __init__(
    self,
    *,
    target_list: Optional[List[str]] = None,
    is_colored: bool = False,
    times: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        movie_list (List[str], optional): list of movies. Defaults to None.
        is_colored (bool, optional):  flag to output in color. Defaults to False.
        times (Tuple[float, float], optional): [start, stop] parameters for trimming movie (s). If this variable is None, this will be selected using GUI window Defaults to None.
    """
    self.__target_list = target_list
    self.__colored = is_colored
    self.__times = times

  def __get_target_list(self) -> Optional[List[str]]:
    return self.__target_list

  def __is_colored(self) -> bool:
    return self.__colored

  def __get_times(self) -> Optional[Tuple[float, float]]:
    return self.__times

  def execute(self):
    """trimming process

    Returns:
        Optional[List[str]]: if process is not executed, None is returned. list of output (movie) path names
    """
    if self.__get_times() is not None:
      time = self.__get_times()
      if time[1] <= time[0]:
        print("stop must be > start")
        return None

    output_path_list = get_output_path(self.__get_target_list(), "trimmed")
    return_list: List[str] = []

    for movie, output_path in zip(self.__get_target_list(), output_path_list):

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)

      if self.__get_times() is None:
        time = self.__select_times(str(output_path), frames, fps, cap)

      if time is None:
        continue

      f1, f2 = round(time[0] * fps), round(time[1] * fps)

      if frames < f1 or frames < f2:
        print("start & stop ({0},{1}) must be < max frame ({2})".format(f1, f2, frames))
        continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (W, H), self.__is_colored())
      print("trimming movie '{0}'...".format(movie))

      while f1 <= f2:

        ret, frame = cap.read()
        if not ret:
          break
        f1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        f = frame if self.__is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output.write(f)

    return return_list if return_list else None

  def __select_times(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[float, float]]:
    """select(get) parametes for trimming using GUI window

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[float, float]]: [start, stop] times for trimming movie (s).
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists, is_start_on, is_stop_on = False, False, False
    start_time, stop_time = 0.0, 0.0

    print("--- trim ---\nselect time (start, stop) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    create_frame_trackbars(movie, frames)
    create_start_bool_trackbar(movie)
    create_stop_bool_trackbar(movie)

    while True:

      tgt_frame = get_value_from_frame_trackbars(movie, frames)

      is_start_on, is_bool_changed = get_values_from_start_trackbar(movie, is_start_on)
      if is_bool_changed:
        start_time = tgt_frame / fps if is_start_on else 0.0

      is_stop_on, is_bool_changed = get_values_from_stop_trackbar(movie, is_stop_on)
      if is_bool_changed:
        stop_time = tgt_frame / fps if is_stop_on else 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if help_exists:
        add_texts_upper_left(
          img,
          [
            "[trim]",
            "select start,stop",
            "s: save",
            "q/esc: abort",
            "now: {0:.2f}s".format(tgt_frame / fps),
            "start: {0:.2f}s".format(start_time),
            "stop: {0:.2f}s".format(stop_time),
          ],
        )

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed.")
        print("start,stop ({0},{1})".format(start_time, stop_time))

        if (not is_start_on) or (not is_stop_on):
          print("start or stop is not selected!")
          continue
        if stop_time <= start_time:
          print("stop must be > start!")
          continue

        cv2.destroyAllWindows()
        return (start_time, stop_time)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if press_q_or_Esc(k):
          None
