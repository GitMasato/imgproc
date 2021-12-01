"""process module containing image process functions
"""
import cv2
import imghdr
import math
import numpy
import pathlib
from abc import ABCMeta, abstractmethod
import matplotlib

matplotlib.use('tkagg')
from matplotlib import pyplot
from typing import List, Optional, Tuple


def sort_target_type(target_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
  """sort input files by type

  Args:
      target_list (List[str]): list of pictures or movies or directories where pictures are stored

  Returns:
      Tuple[List[str], List[str], List[str]]: list of movies, pictures, and directories
  """
  movie_list, picture_list, directory_list = [], [], []

  for target in target_list:

    if pathlib.Path(target).is_dir():
      directory_list.append(target)

    elif pathlib.Path(target).is_file():

      if imghdr.what(target) is not None:
        picture_list.append(target)
      else:
        movie_list.append(target)

  return (movie_list, picture_list, directory_list)


def no(no):
  """call back function and meaningless"""
  pass


class ABCProcess(metaclass=ABCMeta):
  """abstract base class for image processing class"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
    """
    self.__target_list = target_list
    self.__colored = is_colored

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """image process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_target_list(self) -> List[str]:
    return self.__target_list

  def _is_colored(self) -> bool:
    return self.__colored

  def _get_movie_info(self, cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
    """get movie information

    Args:
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Tuple[int, int, int, float]: W, H, total frame, fps of movie
    """
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
    return (int(W), int(H), frames, fps)

  def _get_output_path(self, target_list: List[str], output: str) -> List[pathlib.Path]:
    """get output path list

    Args:
        target_list (List[str]): list of pictures or movies or directories where pictures are stored
        output (str): name of the lowest level directory (without path)

    Returns:
        List[pathlib.Path]: list of pathlib.Path object. Path specifies where outputs are created
    """
    output_path_list: List[pathlib.Path] = []

    for target in target_list:

      target_path = pathlib.Path(target).resolve()
      layers = target_path.parts

      if not target_path.is_dir() and not target_path.is_file():
        print("'{0}' does not exist!".format(str(target_path)))
        continue

      if "cv2" in layers:
        p_path = target_path.parents[1] if target_path.is_file() else target_path.parent
        output_path_list.append(p_path / output)
      else:
        p = target_path.stem if target_path.is_file() else target_path.name
        output_path_list.append(pathlib.Path.cwd() / "cv2" / p / output)

    return output_path_list

  def _create_output_directory(self, output: str):
    """create output directory

    Args:
        output (str): name of the lowest level directory (without path)
    """
    output_path = pathlib.Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

  def _press_q_or_Esc(self, key_input: int) -> bool:
    """check if to press q or Esq

    Returns:
        bool: if to press q or Esq. Being pressed is True
    """
    if (key_input == ord("q")) or (key_input == 27):
      cv2.destroyAllWindows()
      print("'q' or 'Esc' is pressed. abort")
      return True
    else:
      return False

  def _create_frame_trackbars(self, cv2_window: str, input_frames: int):
    """createTrackbar 'frame\n' and 'frame s\n' trackbars for cv2 GUI"""
    tick = 100 if 100 < input_frames else input_frames
    cv2.createTrackbar("frame\n", cv2_window, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", cv2_window, 0, (int)(input_frames / 100) + 1, no)

  def _read_frame_trackbars(self, cv2_window: str, input_frames: int) -> int:
    """read values from 'frame\n' and 'frame s\n' trackbars

    Returns:
        int: target frame number
    """
    division_number = (int)(input_frames / 100) + 1
    frame = cv2.getTrackbarPos("frame\n", cv2_window) * division_number
    frame_s = cv2.getTrackbarPos("frame s\n", cv2_window)
    tgt_frame = frame + frame_s if frame + frame_s < input_frames else input_frames
    return tgt_frame

  def _read_bool_trackbar(self, cv2_window: str, trackbar: str,
                          is_trackbar_on: bool) -> Tuple[bool, bool]:
    """read bool values ([bool status, if status is changed]) from bool trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    if not is_trackbar_on:
      if cv2.getTrackbarPos(trackbar, cv2_window):
        return (True, True)
      else:
        return (False, False)
    else:
      if not cv2.getTrackbarPos(trackbar, cv2_window):
        return (False, True)
      else:
        return (True, False)

  def _add_texts_upper_left(self, img: numpy.array, texts: List[str]):
    """add texts into upper left corner of cv2 object

    Args:
        img (numpy.array): cv2 image object
        texts (List[str]): texts
    """
    position = (5, 0)

    for id, text in enumerate(texts):
      pos = (position[0], position[1] + 30 * (id + 1))
      cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 10)
      cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)


class ABCAnimatingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of animating process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__fps = fps

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """animating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_fps(self) -> Optional[float]:
    return self.__fps

  def _create_fps_trackbar(self, cv2_window: str):
    """create 'fps\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("fps\n", cv2_window, 10, 50, no)

  def _read_fps_trackbar(self, cv2_window: str) -> float:
    """read value from 'fps\n' trackbar

    Returns:
        float: fps
    """
    fps_read = cv2.getTrackbarPos("fps\n", cv2_window)
    return 1.0 if fps_read == 0 else float(fps_read)


class AnimatingMovie(ABCAnimatingProcess):
  """class to create animation (movie) from pictures"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, fps=fps)

  def execute(self) -> Optional[List[str]]:
    """animating process to create movie

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "animated")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_fps() is not None:
        tgt_fps = self._get_fps()
      else:
        tgt_fps = self.__select_fps(output_name, frames, cap)

      if tgt_fps is None:
        tgt_fps = fps

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, tgt_fps, (W, H), self._is_colored())
      print("animating movies...")

      while True:

        ret, frame = cap.read()
        if not ret:
          break
        f = frame if self._is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output.write(f)

    return return_list if return_list else None

  def __select_fps(self, movie: str, frames: int,
                   cap: cv2.VideoCapture) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[float]: fps
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)

    print("--- animate ---\nselect fps in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, frames)
    self._create_fps_trackbar(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      fps = self._read_fps_trackbar(movie)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if help_exists:
        h = ["[animate]", "select fps", "s:save", "h:on/off help", "q/esc:abort"]
        h.extend(["frame: {0}".format(tgt_frame), "fps: {0:.2f}".format(fps)])
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. fps is saved ({0:.2f})".format(fps))
        cv2.destroyAllWindows()
        return fps

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ABCAnimatingPictureProcess(ABCAnimatingProcess, metaclass=ABCMeta):
  """abstract base class of animating picture process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, fps=fps)

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """animating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _select_fps(self, movie: str, picture_list: List[str]) -> Optional[float]:
    """select(get) fps for animating pictures using GUI window

    Args:
        movie (str): movie name
        picture_list (List[str]): picture list

    Returns:
        Optional[float]: fps of movie
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)

    print("--- animate ---\nselect fps in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, len(picture_list) - 1)
    self._create_fps_trackbar(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, len(picture_list) - 1)
      fps = self._read_fps_trackbar(movie)
      img = cv2.imread(picture_list[tgt_frame])

      if help_exists:
        h = ["[animate]", "select fps", "s:save", "h:on/off help", "q/esc:abort"]
        h.extend(["now: {0}".format(tgt_frame), "fps: {0:.2f}".format(fps)])
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. fps is saved ({0:.2f})".format(fps))
        cv2.destroyAllWindows()
        return fps

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None

  def _create_movie(self, picture_list: List[str], output_name: str) -> Optional[str]:
    """create movie from pictures

    Args:
        picture_list (List[str]): list of pictures
        output_name (str): movie name

    Returns:
        Optional[str]: created movie name if that is created
    """
    if self._get_fps() is not None:
      fps = self._get_fps()
    else:
      fps = self._select_fps(output_name, picture_list)

    if fps is None:
      print("animate: abort!")
      return None

    W_list, H_list = [], []

    for picture in picture_list:
      img = cv2.imread(picture)
      W_list.append(img.shape[1])
      H_list.append(img.shape[0])

    W, H = max(W_list), max(H_list)
    self._create_output_directory(output_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(output_name, fourcc, fps, (W, H), self._is_colored())
    print("animating pictures ({0})".format(output_name))

    for picture in picture_list:

      img = cv2.imread(picture)
      f1 = numpy.zeros((H, W, 3), numpy.uint8)
      f1[:img.shape[0], :img.shape[1]] = img[:]
      f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      output.write(f2)

    return output_name


class AnimatingPicture(ABCAnimatingPictureProcess):
  """class to create animation (movie) from pictures"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, fps=fps)

  def execute(self) -> Optional[List[str]]:
    """animating process to create movie

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pictures = self._get_target_list()
    output_path_list = self._get_output_path(pictures, "animated")
    name = str(output_path_list[0] / pathlib.Path(pictures[0]).stem) + ".mp4"
    movie = self._create_movie(pictures, name)
    return None if movie is None else [movie]


class AnimatingPictureDirectory(ABCAnimatingPictureProcess):
  """class to create animation (movie) from pictures"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      fps: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, fps=fps)

  def execute(self) -> Optional[List[str]]:
    """animating process to create movie

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "animated")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      output_name = str(output_path / directory_path.name) + ".mp4"
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      movie = self._create_movie(p_list, output_name)

      if movie is not None:
        return_list.append(movie)

    return return_list if return_list else None


class ABCBinarizingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of binarizing process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=False)
    self.__thresholds = thresholds

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """binarizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def _create_threshold_trackbars(self, cv2_window: str):
    """create 'low\n', 'high\n' trackbars for cv2 GUI"""
    cv2.createTrackbar("low\n", cv2_window, 0, 255, no)
    cv2.createTrackbar("high\n", cv2_window, 255, 255, no)

  def _read_threshold_trackbars(self, cv2_window: str) -> Tuple[int, int]:
    """read values from 'low\n', 'high\n' trackbars

    Returns:
        Tuple[int, int]: thresholds low and high
    """
    low = cv2.getTrackbarPos("low\n", cv2_window)
    high = cv2.getTrackbarPos("high\n", cv2_window)
    return (low, high)


class BinarizingMovie(ABCBinarizingProcess):
  """class to binarize movie
  """

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, thresholds=thresholds)

  def execute(self) -> Optional[List[str]]:
    """binarizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "binarized")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_thresholds() is not None:
        thresholds = self._get_thresholds()
      else:
        thresholds = self.__select_thresholds(output_name, frames, fps, cap)

      if thresholds is None:
        continue

      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      self._create_output_directory(output_name)
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

  def __select_thresholds(self, movie: str, frames: int, fps: float,
                          cap: cv2.VideoCapture) -> Optional[Tuple[int, int]]:
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

    print("--- binarize ---\nselect threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, frames)
    self._create_threshold_trackbars(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      low, high = self._read_threshold_trackbars(movie)

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        h = ["[binarize]", "select thresholds", "s:save"]
        h.extend(["h:on/off help", "q/esc:abort"])
        h.extend(["now: {0:.2f}s".format(tgt_frame / fps)])
        h.extend(["low: {0}".format(low), "high: {0}".format(high)])
        self._add_texts_upper_left(bin_2, h)

      cv2.imshow(movie, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. threshold ({0}, {1})".format(low, high))

        if high <= low:
          print("high luminance threshold must be > low")
          continue
        else:
          cv2.destroyAllWindows()
          return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class BinarizingPicture(ABCBinarizingProcess):
  """class to binarize picture
  """

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, thresholds=thresholds)

  def execute(self) -> Optional[List[str]]:
    """binarizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pictures = self._get_target_list()
    output_path_list = self._get_output_path(pictures, "binarized")
    return_list: List[str] = []

    for picture, output_path in zip(pictures, output_path_list):

      picture_path = pathlib.Path(picture)
      name = str(output_path / picture_path.name)
      img = cv2.imread(picture)

      if self._get_thresholds() is not None:
        thresholds = self._get_thresholds()
      else:
        thresholds = self.__select_thresholds(name, img)

      if thresholds is None:
        continue

      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        continue

      print("binarizing picture '{0}'...".format(picture))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
      return_list.append(name)
      self._create_output_directory(name)
      cv2.imwrite(name, bin_2)

    return return_list if return_list else None

  def __select_thresholds(self, picture: str,
                          img: numpy.array) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        picture (str): name of image
        img (numpy.array): cv2 image object

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)

    print("--- binarize ---\nselect threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_threshold_trackbars(picture)

    while True:

      low, high = self._read_threshold_trackbars(picture)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        h = ["[binarize]", "select thresholds", "s:save"]
        h.extend(["h:on/off help", "q/esc:abort"])
        h.extend(["low: {0}".format(low), "high: {0}".format(high)])
        self._add_texts_upper_left(bin_2, h)

      cv2.imshow(picture, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. threshold ({0}, {1})".format(low, high))

        if high <= low:
          print("high luminance threshold must be > low")
          continue
        else:
          cv2.destroyAllWindows()
          return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class BinarizingPictureDirectory(ABCBinarizingProcess):
  """class to binarize picture
  """

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, thresholds=thresholds)

  def execute(self) -> Optional[List[str]]:
    """binarizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "binarized")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self._get_thresholds() is not None:
        thresholds = self._get_thresholds()
      else:
        thresholds = self.__select_thresholds(str(output_path), p_list)

      if thresholds is None:
        continue

      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        continue

      return_list.append(str(output_path))
      print("binarizing picture in '{0}'...".format(directory))

      for p in p_list:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
        ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
        self._create_output_directory(str(output_path / pathlib.Path(p).name))
        cv2.imwrite(str(output_path / pathlib.Path(p).name), bin_2)

    return return_list if return_list else None

  def __select_thresholds(self, directory: str,
                          picture_list: List[str]) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        directory (str): directory name
        picture_ist (List[str]): picture list

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)

    print("--- binarize ---\nselect threshold (low, high) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(directory, len(picture_list) - 1)
    self._create_threshold_trackbars(directory)

    while True:

      tgt_frame = self._read_frame_trackbars(directory, len(picture_list) - 1)
      low, high = self._read_threshold_trackbars(directory)

      img = cv2.imread(picture_list[tgt_frame])
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        h = ["[binarize]", "select thresholds", "s:save", "h:on/off help"]
        h.extend(["q/esc:abort", "frame: {0}".format(tgt_frame)])
        h.extend(["low: {0}".format(low), "high: {0}".format(high)])
        self._add_texts_upper_left(bin_2, h)

      cv2.imshow(directory, bin_2)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. threshold ({0}, {1})".format(low, high))

        if high <= low:
          print("high luminance threshold must be > low")
          continue
        else:
          cv2.destroyAllWindows()
          return (low, high)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class CapturingMovie(ABCProcess):
  """class to capture movie"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      times: Optional[Tuple[float, float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). If this variable is None, this will be selected using GUI window Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__times = times

  def __get_times(self) -> Optional[Tuple[float, float, float]]:
    return self.__times

  def execute(self) -> Optional[List[str]]:
    """capturing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "captured")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self.__get_times() is not None:
        time = self.__get_times()
      else:
        time = self.__select_times(str(output_path), frames, fps, cap)

      if time is None:
        continue

      if time[1] <= time[0]:
        print("stop ({0}) must be > start ({1})".format(time[1], time[0]))
        continue

      if frames < round(time[1] * fps):
        print("stop frame ({0}) must be < ({1})".format(round(time[1] * fps), frames))
        continue

      if time[1] - time[0] <= time[2]:
        print("difference between stop and start must be > time step")
        continue

      if time[2] < 0.001:
        print("time step must be > 0")
        continue

      capture_time = time[0]
      return_list.append(str(output_path))
      print("capturing movie '{0}'...".format(movie))

      while capture_time <= time[1]:

        cap.set(cv2.CAP_PROP_POS_FRAMES, round(capture_time * fps))
        ret, frame = cap.read()
        if not ret:
          break

        f = frame if self._is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pic_time = int(round(capture_time - time[0], 3) * 1000)
        pic_name = "{0}/{1:08}_ms.png".format(str(output_path), pic_time)
        self._create_output_directory(pic_name)
        cv2.imwrite(pic_name, f)
        capture_time += time[2]

    return return_list if return_list else None

  def __select_times(self, movie: str, frames: int, fps: float,
                     cap: cv2.VideoCapture) -> Optional[Tuple[float, float, float]]:
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

    print("--- capture ---\nselect time (start, stop, step) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists, is_start_on, is_stop_on = False, False, False
    start_time, stop_time = 0.0, 0.0
    self._create_frame_trackbars(movie, frames)
    self.__create_start_trackbar(movie)
    self.__create_stop_trackbar(movie)
    self.__create_step_trackbars(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      time_step = self.__read_step_trackbar(movie)

      is_start_on, is_changed = self.__read_start_trackbar(movie, is_start_on)
      if is_changed:
        start_time = tgt_frame / fps if is_start_on else 0.0

      is_stop_on, is_changed = self.__read_stop_trackbar(movie, is_stop_on)
      if is_changed:
        stop_time = tgt_frame / fps if is_stop_on else 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if help_exists:
        h = [
            "[capture]",
            "select start,stop,step",
        ]
        h.extend(["s: save", "h:on/off help", "q/esc: abort"])
        h.append("now: {0:.2f}s".format(tgt_frame / fps),)
        h.append("start: {0:.2f}s".format(start_time))
        h.append("stop: {0:.2f}s".format(stop_time))
        h.append("step: {0:.2f}s".format(time_step))
        self._add_texts_upper_left(img, h)

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
        if self._press_q_or_Esc(k):
          return None

  def __create_start_trackbar(self, cv2_window: str):
    """create 'start cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("start cap\n", cv2_window, 0, 1, no)

  def __read_start_trackbar(self, cv2_window: str,
                            is_start_on: bool) -> Tuple[bool, bool]:
    """read values from 'start cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "start cap\n", is_start_on)

  def __create_stop_trackbar(self, cv2_window: str):
    """create 'stop cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("stop cap\n", cv2_window, 0, 1, no)

  def __read_stop_trackbar(self, cv2_window: str,
                           is_stop_on: bool) -> Tuple[bool, bool]:
    """read values from 'stop cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "stop cap\n", is_stop_on)

  def __create_step_trackbars(self, cv2_window: str):
    """create 'step 10ms\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("step 10ms\n", cv2_window, 100, 100, no)

  def __read_step_trackbar(self, cv2_window: str) -> float:
    """read values from 'step 10ms\n' trackbars

    Returns:
        float: time step
    """
    return cv2.getTrackbarPos("step 10ms\n", cv2_window) * 10 * 0.001


class ABCClippingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of clipping process"""

  def __init__(self,
               *,
               target_list: List[str] = [],
               is_colored: bool = False,
               positions: Optional[Tuple[int, int, int, int]] = None,
               is_y_dir: bool = False):
    """constructor

    Args:
        target_list (List[str]): list of movies, pictures or directories where pictures are stored.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        postisions (Tuple[int, int, int, int], optional): [first_1, first_2, second_1,
        second_2] positions of areas to clip movie/picture. first area must be smaller
        than second area. This Defaults to None. If this variable is None, this will be
        selected using GUI window is_y_dir (bool, optional): flag to clip in y direction. Defaults to False.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__positions = positions
    self.__is_y_dir = is_y_dir

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """clipping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def _draw_clipping_line(self, img: numpy.array, W: int, H: int,
                          positions: Tuple[int, int, int, int]):

    first_1 = positions[0] if positions[0] <= positions[1] else positions[1]
    first_2 = positions[1] if positions[0] <= positions[1] else positions[0]
    second_1 = positions[2] if positions[2] <= positions[3] else positions[3]
    second_2 = positions[3] if positions[2] <= positions[3] else positions[2]

    if self.__is_y_dir is False:
      cv2.line(img, (first_1, 0), (first_1, H - 1), (255, 255, 255), 2)
      cv2.line(img, (first_2, 0), (first_2, H - 1), (255, 255, 255), 2)
      cv2.line(img, (second_1, 0), (second_1, H - 1), (255, 255, 255), 2)
      cv2.line(img, (second_2, 0), (second_2, H - 1), (255, 255, 255), 2)

      if first_1 != first_2:
        cv2.rectangle(img, (first_1, 0), (first_2, H - 1), (0, 0, 0), thickness=-1)
      if second_1 != second_2:
        cv2.rectangle(img, (second_1, 0), (second_2, H - 1), (0, 0, 0), thickness=-1)

    else:
      cv2.line(img, (0, first_1), (W - 1, first_1), (255, 255, 255), 2)
      cv2.line(img, (0, first_2), (W - 1, first_2), (255, 255, 255), 2)
      cv2.line(img, (0, second_1), (W - 1, second_1), (255, 255, 255), 2)
      cv2.line(img, (0, second_2), (W - 1, second_2), (255, 255, 255), 2)

      if first_1 is not first_2:
        cv2.rectangle(img, (0, first_1), (W - 1, first_2), (0, 0, 0), thickness=-1)
      if second_1 is not second_2:
        cv2.rectangle(img, (0, second_1), (W - 1, second_2), (0, 0, 0), thickness=-1)

  def _create_area_trackbars(self, cv2_window: str, W: int, H: int):
    """create 'first_1', 'first_2', 'second_1', 'second_2' trackbars for cv2 GUI"""
    if self.__is_y_dir is False:
      cv2.createTrackbar("first_1", cv2_window, 0, W, no)
      cv2.createTrackbar("first_2", cv2_window, 0, W, no)
      cv2.createTrackbar("second_1", cv2_window, W, W, no)
      cv2.createTrackbar("second_2", cv2_window, W, W, no)
    else:
      cv2.createTrackbar("first_1", cv2_window, 0, H, no)
      cv2.createTrackbar("first_2", cv2_window, 0, H, no)
      cv2.createTrackbar("second_1", cv2_window, H, H, no)
      cv2.createTrackbar("second_2", cv2_window, H, H, no)

  def _read_area_trackbars(self, cv2_window: str) -> Tuple[int, int, int, int]:
    """read values from 'first_1', 'first_2', 'second_1', 'second_2' trackbars

    Returns:
        Tuple[int, int, int, int]: positions of clipping areas
    """
    first_1 = cv2.getTrackbarPos("first_1", cv2_window)
    first_2 = cv2.getTrackbarPos("first_2", cv2_window)
    second_1 = cv2.getTrackbarPos("second_1", cv2_window)
    second_2 = cv2.getTrackbarPos("second_2", cv2_window)
    return first_1, first_2, second_1, second_2

  def _clip(self, img: numpy.array, pos: Tuple[int, int]) -> numpy.array:
    """clip"""

    if pos[0] == pos[1]:
      return img

    if self.__is_y_dir is False:

      if pos[0] == 0:
        return img[0:img.shape[0], pos[1]:img.shape[1]]

      elif pos[1] == img.shape[1]:
        return img[0:img.shape[0], 0:pos[0]]

      else:
        clip_1 = img[0:img.shape[0], 0:pos[0]]
        clip_2 = img[0:img.shape[0], pos[1]:img.shape[1]]
        return cv2.hconcat([clip_1, clip_2])

    else:
      if pos[0] == 0:
        return img[pos[1]:img.shape[0], 0:img.shape[1]]

      elif pos[1] == img.shape[0]:
        return img[0:pos[0], 0:img.shape[1]]

      else:
        clip_1 = img[0:pos[0], 0:img.shape[1]]
        clip_2 = img[pos[1]:img.shape[0], 0:img.shape[1]]
        return cv2.vconcat([clip_1, clip_2])

  def _get_clipped_image(self, img: numpy.array, pos: Tuple[int, int, int,
                                                            int]) -> numpy.array:
    """get clipped image

    Args:
        img (numpy.array): cv2 image object
        pos (Tuple[int, int, int, int]): positions of clipping areas
        pos[2] and pos[3] must be larger than pos[0] and pos[1]

    Returns:
        img (numpy.array): cv2 image object
    """
    first_1 = pos[0] if pos[0] <= pos[1] else pos[1]
    first_2 = pos[1] if pos[0] <= pos[1] else pos[0]
    second_1 = pos[2] if pos[2] <= pos[3] else pos[3]
    second_2 = pos[3] if pos[2] <= pos[3] else pos[2]
    deduction = first_2 - first_1
    clipped_1 = self._clip(img, (first_1, first_2))
    return self._clip(clipped_1, (second_1 - deduction, second_2 - deduction))


class ClippingMovie(ABCClippingProcess):
  """class to clip movie"""

  def __init__(self,
               *,
               target_list: List[str] = [],
               is_colored: bool = False,
               positions: Optional[Tuple[int, int, int, int]] = None,
               is_y_dir: bool = False):
    """constructor

    Args:
        target_list (List[str]): list of movies, pictures or directories where pictures are stored.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        postisions (Tuple[int, int, int, int], optional): [first_1, first_2, second_1,
        second_2] positions of areas to clip movie/picture. first area must be smaller
        than second area. This Defaults to None. If this variable is None, this will be
        selected using GUI window is_y_dir (bool, optional): flag to clip in y direction. Defaults to False.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions,
                     is_y_dir=is_y_dir)

  def execute(self) -> Optional[List[str]]:
    """clipping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "clipped")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(output_name, W, H, frames, fps, cap)

      if pos is None:
        continue

      if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
        print("2nd area must be >= 1st area")
        continue

      ret, frame = cap.read()
      output_H, output_W, null = self._get_clipped_image(frame, pos).shape

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")

      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (W, H), self._is_colored())
      output = cv2.VideoWriter(output_name, fourcc, fps, (output_W, output_H),
                               self._is_colored())
      print("clipping movie '{0}'...".format(movie))

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        f1 = self._get_clipped_image(frame, pos)
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_positions(
      self,
      movie: str,
      W: int,
      H: int,
      frames: int,
      fps: float,
      cap: cv2.VideoCapture,
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) positions for clipping process using GUI window

    Args:
        movie (str): movie file name
        W (int): W video length
        H (int): H video length
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[int, int, int, int]]: [first_1, first_2, second_1, second_2] four positions to clip image
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)

    print("--- clip ---\nselect four positions in GUI window! (2nd area must be > 1st)")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, frames)
    self._create_area_trackbars(movie, W, H)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      pos = self._read_area_trackbars(movie)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      self._draw_clipping_line(img, W, H, pos)

      if help_exists:
        h = ["[clip]", "select areas", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("now: {0:.2f}s".format(tgt_frame / fps))
        h.append("first: [{0:.1f},{1:.1f}]".format(pos[0], pos[1]))
        h.append("second: [{0:.1f},{1:.1f}]".format(pos[2], pos[3]))
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
          print("2nd area must be >= 1st area")
          continue

        else:
          print("'s' is pressed. positions are saved {0}".format(pos))
          cv2.destroyAllWindows()
          return pos

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ClippingPicture(ABCClippingProcess):
  """class to clip picture"""

  def __init__(self,
               *,
               target_list: List[str] = [],
               is_colored: bool = False,
               positions: Optional[Tuple[int, int, int, int]] = None,
               is_y_dir: bool = False):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        postisions (Tuple[int, int, int, int], optional): [first_1, first_2, second_1,
        second_2] positions of areas to clip movie/picture. first area must be smaller
        than second area. This Defaults to None. If this variable is None, this will be
        selected using GUI window is_y_dir (bool, optional): flag to clip in y direction. Defaults to False.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions,
                     is_y_dir=is_y_dir)

  def execute(self) -> Optional[List[str]]:
    """clipping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "clipped")
    return_list: List[str] = []

    for picture, output_path in zip(self._get_target_list(), output_path_list):

      name = str(output_path / pathlib.Path(picture).name)
      img = cv2.imread(picture)

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(name, img)

      if pos is None:
        continue

      if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
        print("2nd area must be >= 1st area")
        continue

      print("clipping picture '{0}'...".format(picture))
      return_list.append(name)
      f1 = self._get_clipped_image(img, pos)
      f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      self._create_output_directory(name)
      cv2.imwrite(name, f2)

    return return_list if return_list else None

  def __select_positions(self, picture: str,
                         img: numpy.array) -> Optional[Tuple[int, int, int, int]]:
    """select(get) positions for clipping process using GUI window

    Args:
        picture (str): image name
        img (numpy.array): cv2 image object

    Returns:
        Optional[Tuple[int, int, int, int]]: [first_1, first_2, second_1, second_2] four positions to clip image
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)

    print("--- clip ---\nselect four positions in GUI window! (2nd area must be > 1st)")
    print("(s: save, h:on/off help, q/esc: abort)")

    W, H = img.shape[1], img.shape[0]
    help_exists = False
    self._create_area_trackbars(picture, W, H)

    while True:

      img_show = img.copy()
      pos = self._read_area_trackbars(picture)
      self._draw_clipping_line(img_show, W, H, pos)

      if help_exists:
        h = ["[clip]", "select areas", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("first: [{0:.1f},{1:.1f}]".format(pos[0], pos[1]))
        h.append("second: [{0:.1f},{1:.1f}]".format(pos[2], pos[3]))
        self._add_texts_upper_left(img, h)

      cv2.imshow(picture, img_show)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
          print("2nd area must be >= 1st area")
          continue

        else:
          print("'s' is pressed. positions are saved {0}".format(pos))
          cv2.destroyAllWindows()
          return pos

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ClippingPictureDirectory(ABCClippingProcess):
  """class to clip picture"""

  def __init__(self,
               *,
               target_list: List[str] = [],
               is_colored: bool = False,
               positions: Optional[Tuple[int, int, int, int]] = None,
               is_y_dir: bool = False):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        postisions (Tuple[int, int, int, int], optional): [first_1, first_2, second_1,
        second_2] positions of areas to clip movie/picture. first area must be smaller
        than second area. This Defaults to None. If this variable is None, this will be
        selected using GUI window is_y_dir (bool, optional): flag to clip in y direction. Defaults to False.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions,
                     is_y_dir=is_y_dir)

  def execute(self) -> Optional[List[str]]:
    """clipping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "clipped")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(str(output_path), p_list)

      if pos is None:
        continue

      if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
        print("2nd area must be >= 1st area")
        continue

      return_list.append(str(output_path))
      print("clipping picture in '{0}'...".format(directory))

      for p in p_list:

        img = cv2.imread(p)
        f1 = self._get_clipped_image(img, pos)
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        self._create_output_directory(str(output_path / pathlib.Path(p).name))
        cv2.imwrite(str(output_path / pathlib.Path(p).name), f2)

    return return_list if return_list else None

  def __select_positions(
      self, directory: str,
      picture_list: List[str]) -> Optional[Tuple[int, int, int, int]]:
    """select(get) positions for clipping process using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[Tuple[int, int, int, int]]: [first_1, first_2, second_1, second_2] four positions to clip image
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)

    print("--- clip ---\nselect four positions in GUI window! (2nd area must be > 1st)")
    print("(s: save, h:on/off help, q/esc: abort)")

    H, W, null = cv2.imread(picture_list[0]).shape
    help_exists = False
    self._create_frame_trackbars(directory, len(picture_list) - 1)
    self._create_area_trackbars(directory, W, H)

    while True:

      tgt_frame = self._read_frame_trackbars(directory, len(picture_list) - 1)
      img = cv2.imread(picture_list[tgt_frame])
      pos = self._read_area_trackbars(directory)
      self._draw_clipping_line(img, W, H, pos)

      if help_exists:
        h = ["[clip]", "select areas", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("first: [{0:.1f},{1:.1f}]".format(pos[0], pos[1]))
        h.append("second: [{0:.1f},{1:.1f}]".format(pos[2], pos[3]))
        self._add_texts_upper_left(img, h)

      cv2.imshow(directory, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        if pos[3] < pos[1] or pos[3] < pos[0] or pos[2] < pos[1] or pos[2] < pos[0]:
          print("2nd area must be >= 1st area")
          continue

        else:
          print("'s' is pressed. positions are saved {0}".format(pos))
          cv2.destroyAllWindows()
          return pos

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ABCConcatenatingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of concatenating process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      number_x: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        number_x (Optional[int], optional): number of targets concatenated in x direction. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__number_x = number_x

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """concatenating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_number_x(self) -> Optional[int]:
    return self.__number_x

  def _create_x_trackbars(self, cv2_window: str):
    """create 'x\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("x\n", cv2_window, 1, 5, no)

  def _read_x_trackbar(self, cv2_window: str, total_movies_number: int) -> int:
    """read values from 'x\n' trackbars

    Args:
        cv2_window (str): name of cv GUI window
        total_movies_number (int): number of total movies

    Returns:
        int: number of targets concatenated in x direction
    """
    if cv2.getTrackbarPos("x\n", cv2_window) == 0:
      return 1
    elif total_movies_number < cv2.getTrackbarPos("x\n", cv2_window):
      return total_movies_number
    else:
      return cv2.getTrackbarPos("x\n", cv2_window)

  def _vconcat_H(self, img_list: List[numpy.array], W: int) -> numpy.array:
    """concatenate images in H direction

    Args:
        img_list (List[numpy.array]): list of images
        W (int): W size

    Returns:
        numpy.array: concatenated image
    """
    concat_list = []

    for img in img_list:
      size = (W, int(img.shape[0] * W / img.shape[1]))
      concat_list.append(cv2.resize(img, size, interpolation=cv2.INTER_CUBIC))

    return cv2.vconcat(concat_list)

  def _vconcat_W(self, img_list: List[numpy.array], H: int) -> numpy.array:
    """concatenate images in W direction

    Args:
        img_list (List[numpy.array]): list of images
        H (int): H size

    Returns:
        numpy.array: concatenated image
    """
    concat_list = []

    for img in img_list:
      size = (int(H / img.shape[0] * img.shape[1]), H)
      concat_list.append(cv2.resize(img, size, interpolation=cv2.INTER_CUBIC))

    return cv2.hconcat(concat_list)


class ConcatenatingMovie(ABCConcatenatingProcess):
  """class to concatenate movie"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      number_x: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        number_x (Optional[int], optional): number of targets concatenated in x direction. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, number_x=number_x)

  def execute(self) -> Optional[List[str]]:
    """resizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    movies = self._get_target_list()
    frame_number_list, fps_list = [], []

    if 25 < len(movies):
      print("'{0}' movies given. max is 25".format(len(movies)))
      return None

    for movie in movies:
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)
      frame_number_list.append(frames)
      fps_list.append(fps)

    if self._get_number_x() is not None:
      number_x = self._get_number_x()
    else:
      number_x = self.__select_number_of_movies_x_dir(movies, frame_number_list)

    if number_x is None:
      return None

    number_y = math.ceil(len(movies) / number_x)
    concat = self.__get_concat_frame(movies, frame_number_list, 0, number_x, number_y)
    size = (concat.shape[1], concat.shape[0])

    output_path_list = self._get_output_path([movies[0]], "concatenated")
    output_name = str(output_path_list[0] / pathlib.Path(movies[0]).stem) + ".mp4"
    self._create_output_directory(output_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(output_name, fourcc, fps_list[0], size, self._is_colored())
    print("concatenating movie '{0}'...".format(output_name))

    for id in range(max(frame_number_list) + 1):
      f1 = self.__get_concat_frame(movies, frame_number_list, id, number_x, number_y)
      f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      output.write(f2)

    return [output_name]

  def __select_number_of_movies_x_dir(
      self,
      movie_list: List[str],
      frame_number_list: List[int],
  ) -> Optional[int]:
    """select(get) number of concatenating using GUI window

    Args:
        movie (List[str]): movie list
        frame_number_list (List[int]): list of frame numbers of movies

    Returns:
        Optional[int]: number of movies concatenated in x direction
    """
    cv2.namedWindow(movie_list[0], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie_list[0], 500, 700)

    print("--- concatenate ---\nselect number of pictures in x-dir in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie_list[0], max(frame_number_list))
    self._create_x_trackbars(movie_list[0])

    while True:

      tgt_frame = self._read_frame_trackbars(movie_list[0], max(frame_number_list))
      number_x = self._read_x_trackbar(movie_list[0], len(movie_list))
      number_y = math.ceil(len(movie_list) / number_x)
      concat = self.__get_concat_frame(movie_list, frame_number_list, tgt_frame,
                                       number_x, number_y)

      if help_exists:
        h = ["[concatenate]", "select x", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("frame: {0}".format(tgt_frame))
        self._add_texts_upper_left(concat, h)

      cv2.imshow(movie_list[0], concat)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. concatenating number x-dir ({0})".format(number_x))
        cv2.destroyAllWindows()
        return number_x

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None

  def __get_concat_frame(
      self,
      movie_list: List[str],
      frame_number_list: List[int],
      tgt_frame: int,
      number_x: int,
      number_y: int,
  ) -> numpy.array:
    """get concatenated frame of movie

    Args:
        movie_list (List[str]): list of movies
        frame_number_list (List[int]): list of frame numbers of movies
        tgt_frame (int): target frame id
        number_x (int): number of movies concatenated in x direction
        number_y (int): number of movies concatenated in x direction

    Returns:
        numpy.array: concatenated image
    """
    black_img_list, pic_list = [], []

    for id, movie in enumerate(movie_list):

      cap = cv2.VideoCapture(movie)
      W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
      black_img_list.append(numpy.zeros((int(H), int(W), 3), numpy.uint8))

      if frame_number_list[id] < tgt_frame:
        pic_list.append(black_img_list[id])
      else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
        ret, img = cap.read()
        pic_list.append(img)

    for id in range(number_x * number_y - len(movie_list)):
      pic_list.append(black_img_list[0])

    multi = [pic_list[y * number_x:y * number_x + number_x] for y in range(number_y)]
    concat_W_list = [self._vconcat_W(one, pic_list[0].shape[0]) for one in multi]
    return self._vconcat_H(concat_W_list, concat_W_list[0].shape[1])


class ABCConcatenatingPictureProcess(ABCConcatenatingProcess, metaclass=ABCMeta):
  """abstract base class of concatenating picture process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      number_x: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        number_x (Optional[int], optional): number of targets concatenated in x direction. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, number_x=number_x)

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """concatenating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _select_number_of_pictures_x_dir(
      self,
      picture_list: List[str],
      size_list: List[Tuple[int, int]],
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

    print("--- concatenate ---\nselect number of pictures in x-dir in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_x_trackbars(window)

    while True:

      number_x = self._read_x_trackbar(window, len(picture_list))
      number_y = math.ceil(len(picture_list) / number_x)
      concat = self._get_concat_picture(picture_list, number_x, number_y)

      if help_exists:
        h = ["[concatenate]", "select x", "s:save", "h:on/off help", "q/esc:abort"]
        self._add_texts_upper_left(concat, h)

      cv2.imshow(window, concat)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. concatenating number x-dir ({0})".format(number_x))
        cv2.destroyAllWindows()
        return number_x

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None

  def _create_concat_picture(
      self,
      picture_list: List[str],
      output_name: str,
  ) -> Optional[str]:
    """create concatenated picture

    Args:
        picture_list (List[str]): list of pictures
        output_name (str): name of concatenated picture

    Returns:
        Optional[str]: concatenated picture name if that is created
    """
    if 25 < len(picture_list):
      print("'{0}' pictures given. max is 25".format(len(picture_list)))
      return None

    size_list = []

    for picture in picture_list:
      img = cv2.imread(picture)
      size_list.append((img.shape[1], img.shape[0]))

    if self._get_number_x() is not None:
      number_x = self._get_number_x()
    else:
      number_x = self._select_number_of_pictures_x_dir(picture_list, size_list)

    if number_x is None:
      return None

    print("concatenating picture '{0}'...".format(output_name))
    number_y = math.ceil(len(picture_list) / number_x)
    f1 = self._get_concat_picture(picture_list, number_x, number_y)
    f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    self._create_output_directory(output_name)
    cv2.imwrite(output_name, f2)
    return output_name

  def _get_concat_picture(
      self,
      picture_list: List[str],
      number_x: int,
      number_y: int,
  ) -> numpy.array:
    """get concatenated picture

    Args:
        picture_list (List[str]): list of pictures
        number_x (int): number of pictures concatenated in x direction
        number_y (int): number of pictures concatenated in y direction

    Returns:
        numpy.array: concatenated picture
    """
    img_list = [cv2.imread(picture) for picture in picture_list]

    for id in range(number_x * number_y - len(picture_list)):
      black = numpy.zeros((img_list[0].shape[0], img_list[0].shape[1], 3), numpy.uint8)
      img_list.append(black)

    multi = [img_list[y * number_x:y * number_x + number_x] for y in range(number_y)]
    concat_W_list = [self._vconcat_W(one, img_list[0].shape[0]) for one in multi]
    return self._vconcat_H(concat_W_list, concat_W_list[0].shape[1])


class ConcatenatingPicture(ABCConcatenatingPictureProcess):
  """class to concatenate picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      number_x: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        number_x (Optional[int], optional): number of targets concatenated in x direction. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, number_x=number_x)

  def execute(self) -> Optional[List[str]]:
    """concatenating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pictures = self._get_target_list()
    output_path_list = self._get_output_path([pictures[0]], "concatenated")
    name = str(output_path_list[0] / pathlib.Path(pictures[0]).name)
    concat_name = self._create_concat_picture(pictures, name)
    return None if concat_name is None else [concat_name]


class ConcatenatingPictureDirectory(ABCConcatenatingPictureProcess):
  """class to concatenate picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      number_x: Optional[int] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        number_x (Optional[int], optional): number of targets concatenated in x direction. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, number_x=number_x)

  def execute(self) -> Optional[List[str]]:
    """concatenating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "concatenated")
    return_list: List[str] = []

    for idx, directory in enumerate(self._get_target_list()):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no picture exists in '{0}'!".format(directory))
        continue

      file_name = directory_path.name + pathlib.Path(p_list[0]).suffix
      name = str(output_path_list[idx] / file_name)
      concat_name = self._create_concat_picture(p_list, name)

      if concat_name is not None:
        return_list.append(concat_name)

    return return_list if return_list else None


class ABCCroppingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of cropping process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__positions = positions

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """cropping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def _draw_cropping_line(self, img: numpy.array, W: int, H: int,
                          points: List[Tuple[int, int]]):
    if len(points) == 1:
      cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), (255, 255, 255), 2)
      cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), (255, 255, 255), 2)

    elif len(points) == 2:
      cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), (255, 255, 255), 2)
      cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), (255, 255, 255), 2)
      cv2.line(img, (points[1][0], 0), (points[1][0], H - 1), (255, 255, 255), 2)
      cv2.line(img, (0, points[1][1]), (W - 1, points[1][1]), (255, 255, 255), 2)

  def _mouse_on_select_cropping_position(self, event, x, y, flags, params):
    """call back function on mouse click
    """
    points = params
    if event == cv2.EVENT_LBUTTONUP:
      points.append([x, y])
      if len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
      elif len(points) == 3:
        points.clear()


class CroppingMovie(ABCCroppingProcess):
  """class to crop movie"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions)

  def execute(self) -> Optional[List[str]]:
    """cropping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "cropped")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(output_name, W, H, frames, fps, cap)

      if pos is None:
        continue

      if pos[2] <= pos[0] or pos[3] <= pos[1]:
        print("2nd position must be > 1st")
        continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      size = (int(pos[2] - pos[0]), int(pos[3] - pos[1]))
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self._is_colored())
      print("cropping movie '{0}'...".format(movie))

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        f1 = frame[pos[1]:pos[3], pos[0]:pos[2]]
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_positions(
      self,
      movie: str,
      W: int,
      H: int,
      frames: int,
      fps: float,
      cap: cv2.VideoCapture,
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
        Optional[Tuple[int, int, int, int]]: [x_1, y_1, x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)

    print("--- crop ---\nselect two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    help_exists = False
    points: List[Tuple[int, int]] = []
    cv2.setMouseCallback(movie, self._mouse_on_select_cropping_position, points)
    self._create_frame_trackbars(movie, frames)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      self._draw_cropping_line(img, W, H, points)

      if help_exists:
        h = ["[crop]", "select two positions", "frame: {0}".format(tgt_frame)]
        h.extend(["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"])
        h.extend(["{0}".format(p) for p in points])
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. selected positions ({0})".format(points))

        if len(points) == 2:
          print("cropped positions are saved")
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
        if self._press_q_or_Esc(k):
          return None


class CroppingPicture(ABCCroppingProcess):
  """class to crop picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions)

  def execute(self) -> Optional[List[str]]:
    """cropping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "cropped")
    return_list: List[str] = []

    for picture, output_path in zip(self._get_target_list(), output_path_list):

      name = str(output_path / pathlib.Path(picture).name)
      img = cv2.imread(picture)

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(name, img)

      if pos is None:
        continue

      if pos[2] <= pos[0] or pos[3] <= pos[1]:
        print("2nd position must be > 1st")
        continue

      print("cropping picture '{0}'...".format(picture))
      return_list.append(name)
      f1 = img[pos[1]:pos[3], pos[0]:pos[2]]
      f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      self._create_output_directory(name)
      cv2.imwrite(name, f2)

    return return_list if return_list else None

  def __select_positions(self, picture: str,
                         img: numpy.array) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for cropping process using GUI window

    Args:
        picture (str): image name
        img (numpy.array): cv2 image object

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)

    print("--- crop ---\nselect two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    W, H = img.shape[1], img.shape[0]
    points: List[Tuple[int, int]] = []
    help_exists = False
    cv2.setMouseCallback(picture, self._mouse_on_select_cropping_position, points)

    while True:

      img_show = img.copy()
      self._draw_cropping_line(img_show, W, H, points)

      if help_exists:
        h = ["[crop]", "select two positions"]
        h.extend(["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"])
        h.extend(["{0}".format(p) for p in points])
        self._add_texts_upper_left(img, h)

      cv2.imshow(picture, img_show)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. selected positions ({0})".format(points))

        if len(points) == 2:
          print("cropped positions are saved")
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
        if self._press_q_or_Esc(k):
          return None


class CroppingPictureDirectory(ABCCroppingProcess):
  """class to crop picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list,
                     is_colored=is_colored,
                     positions=positions)

  def execute(self) -> Optional[List[str]]:
    """cropping process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "cropped")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self._get_positions() is not None:
        pos = self._get_positions()
      else:
        pos = self.__select_positions(str(output_path), p_list)

      if pos is None:
        continue

      if pos[2] <= pos[0] or pos[3] <= pos[1]:
        print("2nd position must be > 1st")
        continue

      return_list.append(str(output_path))
      print("cropping picture in '{0}'...".format(directory))

      for p in p_list:

        img = cv2.imread(p)
        f1 = img[pos[1]:pos[3], pos[0]:pos[2]]
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        self._create_output_directory(str(output_path / pathlib.Path(p).name))
        cv2.imwrite(str(output_path / pathlib.Path(p).name), f2)

    return return_list if return_list else None

  def __select_positions(
      self, directory: str,
      picture_list: List[str]) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for cropping process using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)

    print("--- crop ---\nselect two positions in GUI window! (2nd must be > 1st)")
    print("(s: save, h:on/off help, c: clear, click: select, q/esc: abort)")

    help_exists = False
    points: List[Tuple[int, int]] = []
    cv2.setMouseCallback(directory, self._mouse_on_select_cropping_position, points)
    self._create_frame_trackbars(directory, len(picture_list) - 1)

    while True:

      tgt_frame = self._read_frame_trackbars(directory, len(picture_list) - 1)
      img = cv2.imread(picture_list[tgt_frame])
      self._draw_cropping_line(img, img.shape[1], img.shape[0], points)

      if help_exists:
        h = ["[crop]", "select two positions", "frame: {0}".format(tgt_frame)]
        h.extend(["s:save", "h:on/off help", "c:clear", "click:select", "q/esc:abort"])
        h.extend(["{0}".format(p) for p in points])
        self._add_texts_upper_left(img, h)

      cv2.imshow(directory, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. selected positions ({0})".format(points))

        if len(points) == 2:
          print("cropped positions are saved")
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
        if self._press_q_or_Esc(k):
          return None


class ABCCreatingLuminanceHistgramProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of creating luminance histgram"""

  def __init__(self, *, target_list: List[str] = [], is_colored: bool = False):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """creating luminance histgram

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _create_color_figure_luminance_histgram(self, picture: str,
                                              output_path: pathlib.Path):
    """create output figure of luminance histgram  in color

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
    self._create_output_directory(str(output_path / pathlib.Path(picture).name))
    fig.savefig(str(output_path / pathlib.Path(picture).name))
    pyplot.close(fig)

  def _create_gray_figure_luminance_histgram(self, picture: str,
                                             output_path: pathlib.Path):
    """create output figure of luminance histgram in gray

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
    self._create_output_directory(str(output_path / pathlib.Path(picture).name))
    fig.savefig(str(output_path / pathlib.Path(picture).name))
    pyplot.close(fig)


class CreatingLuminanceHistgramPicture(ABCCreatingLuminanceHistgramProcess):
  """class to create luminance histgram of picture"""

  def __init__(self, *, target_list: List[str] = [], is_colored: bool = False):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)

  def execute(self) -> Optional[List[str]]:
    """creating luminance histglam

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(),
                                             "histgram_luminance")
    return_list: List[str] = []

    for picture, output_path in zip(self._get_target_list(), output_path_list):

      print("creating luminance histgram of picture '{0}'...".format(picture))
      return_list.append(str(output_path / pathlib.Path(picture).name))

      if self._is_colored():
        self._create_color_figure_luminance_histgram(picture, output_path)
      else:
        self._create_gray_figure_luminance_histgram(picture, output_path)

    return return_list if return_list else None


class CreatingLuminanceHistgramPictureDirectory(ABCCreatingLuminanceHistgramProcess):
  """class to create luminance histgram of picture"""

  def __init__(self, *, target_list: List[str] = [], is_colored: bool = False):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)

  def execute(self) -> Optional[List[str]]:
    """creating luminance histglam

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(),
                                             "histgram_luminance")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      return_list.append(str(output_path))
      print("creating luminance histgram of picture in '{0}'...".format(directory))

      if self._is_colored():
        for p in p_list:
          self._create_color_figure_luminance_histgram(p, output_path)
      else:
        for p in p_list:
          self._create_gray_figure_luminance_histgram(p, output_path)

    return return_list if return_list else None


class ABCResizingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of resizing process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__scales = scales

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """resizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def _create_scale_trackbars(self, cv2_window: str):
    """create 'scale x\n*0.1', 'scale x\n*1.0', 'scale y\n*0.1', 'scale y\n*1.0' trackbars for cv2 GUI"""
    cv2.createTrackbar("scale x\n*0.1", cv2_window, 10, 10, no)
    cv2.createTrackbar("scale x\n*1.0", cv2_window, 1, 10, no)
    cv2.createTrackbar("scale y\n*0.1", cv2_window, 10, 10, no)
    cv2.createTrackbar("scale y\n*1.0", cv2_window, 1, 10, no)

  def _read_scale_trackbars(self, cv2_window: str) -> Tuple[float, float]:
    """read values from 'scale x\n*0.1', 'scale x\n*1.0', 'scale y\n*0.1', 'scale y\n*1.0' trackbars

    Returns:
        Tuple[float, float]: scaling factors x and y directions
    """
    s_x_01 = cv2.getTrackbarPos("scale x\n*0.1", cv2_window)
    s_x_10 = cv2.getTrackbarPos("scale x\n*1.0", cv2_window)
    s_y_01 = cv2.getTrackbarPos("scale y\n*0.1", cv2_window)
    s_y_10 = cv2.getTrackbarPos("scale y\n*1.0", cv2_window)
    s_x = (1 if s_x_01 == 0 else s_x_01) * 0.1 * (1 if s_x_10 == 0 else s_x_10)
    s_y = (1 if s_y_01 == 0 else s_y_01) * 0.1 * (1 if s_y_10 == 0 else s_y_10)
    return s_x, s_y


class ResizingMovie(ABCResizingProcess):
  """class to resize movie"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, scales=scales)

  def execute(self) -> Optional[List[str]]:
    """resizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "resized")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_scales() is not None:
        scales = self._get_scales()
      else:
        scales = self.__select_scales(output_name, W, H, frames, fps, cap)

      if scales is None:
        continue

      size = (int(W * scales[0]), int(H * scales[1]))
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self._is_colored())
      print("resizing movie '{0}'...".format(movie))

      while True:

        ret, frame = cap.read()
        if not ret:
          break
        f1 = cv2.resize(frame, dsize=size)
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_scales(self, movie: str, W: int, H: int, frames: int, fps: float,
                      cap: cv2.VideoCapture) -> Optional[Tuple[float, float]]:
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

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, frames)
    self._create_scale_trackbars(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      s_x, s_y = self._read_scale_trackbars(movie)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        h = ["[resize]", "select scales", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("now: {0:.2f}s".format(tgt_frame / fps))
        h.append("scale: {0:.1f},{1:.1f}".format(s_x, s_y))
        self._add_texts_upper_left(resized, h)

      cv2.imshow(movie, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ResizingPicture(ABCResizingProcess):
  """class to resize picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, scales=scales)

  def execute(self) -> Optional[List[str]]:
    """resizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "resized")
    return_list: List[str] = []

    for picture, output_path in zip(self._get_target_list(), output_path_list):

      name = str(output_path / pathlib.Path(picture).name)
      img = cv2.imread(picture)
      W, H = img.shape[1], img.shape[0]

      if self._get_scales() is not None:
        scales = self._get_scales()
      else:
        scales = self.__select_scales(name, img, W, H)

      if scales is None:
        continue

      print("resizing picture '{0}'...".format(picture))
      return_list.append(name)
      f1 = cv2.resize(img, dsize=(int(W * scales[0]), int(H * scales[1])))
      f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
      self._create_output_directory(name)
      cv2.imwrite(name, f2)

    return return_list if return_list else None

  def __select_scales(self, picture: str, img: numpy.array, W: int,
                      H: int) -> Optional[Tuple[float, float]]:
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

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_scale_trackbars(picture)

    while True:

      s_x, s_y = self._read_scale_trackbars(picture)
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        h = ["[resize]", "select scales", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("scale: {0:.1f},{1:.1f}".format(s_x, s_y))
        self._add_texts_upper_left(resized, h)

      cv2.imshow(picture, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ResizingPictureDirectory(ABCResizingProcess):
  """class to resize picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to []        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, scales=scales)

  def execute(self) -> Optional[List[str]]:
    """resizing process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "resized")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self._get_scales() is not None:
        scales = self._get_scales()
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
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        self._create_output_directory(str(output_path / pathlib.Path(p).name))
        cv2.imwrite(str(output_path / pathlib.Path(p).name), f2)

    return return_list if return_list else None

  def __select_scales(self, directory: str,
                      picture_list: List[str]) -> Optional[Tuple[float, float]]:
    """select(get) resizing scales using GUI window

    Args:
        directory (str): directory name
        picture_list (List[str]): picture list

    Returns:
        Optional[float]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)

    print("--- resize ---\nselect scales (x, y) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(directory, len(picture_list) - 1)
    self._create_scale_trackbars(directory)

    while True:

      tgt_frame = self._read_frame_trackbars(directory, len(picture_list) - 1)
      s_x, s_y = self._read_scale_trackbars(directory)
      img = cv2.imread(picture_list[tgt_frame])
      W, H = img.shape[1], img.shape[0]
      resized = cv2.resize(img, dsize=(int(W * s_x), int(H * s_y)))

      if help_exists:
        h = ["[resize]", "select scales", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("frame: {0}".format(tgt_frame))
        h.append("scale: {0:.1f},{1:.1f}".format(s_x, s_y))
        self._add_texts_upper_left(resized, h)

      cv2.imshow(directory, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales are saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ABCRotatingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of rotating process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__degree = degree

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """rotating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_degree(self) -> Optional[float]:
    return self.__degree

  def _create_degree_trackbars(self, cv2_window: str):
    """create 'degree\n' and 'degree s\n' trackbars for cv2 GUI"""
    cv2.createTrackbar("degree\n", cv2_window, 0, 4, no)
    cv2.createTrackbar("degree s\n", cv2_window, 0, 90, no)

  def _read_degree_trackbars(self, cv2_window: str) -> float:
    """read value from 'degree\n' and 'degree s\n' trackbars

    Returns:
        float: degree
    """
    degree_l = cv2.getTrackbarPos("degree\n", cv2_window) * 90
    degree_s = cv2.getTrackbarPos("degree s\n", cv2_window)
    return degree_l + degree_s

  def _get_rotated_frame_size(self, W: int, H: int, degree: float) -> Tuple[int, int]:
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

  def _get_rotating_affine_matrix(self, center: Tuple[int, int], center_rot: Tuple[int,
                                                                                   int],
                                  degree: float) -> numpy.array:
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

  def _get_rotated_frame(self, img: numpy.array, degree: float) -> numpy.array:
    """get rotated cv2 object

    Args:
        img (numpy.array): cv2 object
        degree (float): degree of rotation

    Returns:
        numpy.array: rotated cv2 object
    """
    W, H = img.shape[1], img.shape[0]
    size_rot = self._get_rotated_frame_size(W, H, degree)
    center_rot = (int(size_rot[0] / 2), int(size_rot[1] / 2))
    a = self._get_rotating_affine_matrix((int(W / 2), int(H / 2)), center_rot, degree)
    return cv2.warpAffine(img, a, size_rot, flags=cv2.INTER_CUBIC)


class RotatingMovie(ABCRotatingProcess):
  """class to rotate movie
  """

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, degree=degree)

  def execute(self) -> Optional[List[str]]:
    """rotating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "rotated")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self._get_degree() is not None:
        degree = self._get_degree()
      else:
        degree = self.__select_degree(output_name, frames, fps, cap)

      if degree is None:
        continue

      size_rot = self._get_rotated_frame_size(W, H, degree)
      center_rot = (int(size_rot[0] / 2), int(size_rot[1] / 2))
      center = (int(W / 2), int(H / 2))
      affine = self._get_rotating_affine_matrix(center, center_rot, degree)

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size_rot, self._is_colored())
      print("rotating movie '{0}'...".format(movie))

      while True:

        ret, frame = cap.read()
        if not ret:
          break
        f1 = cv2.warpAffine(frame, affine, size_rot, flags=cv2.INTER_CUBIC)
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        output.write(f2)

    return return_list if return_list else None

  def __select_degree(self, movie: str, frames: int, fps: float,
                      cap: cv2.VideoCapture) -> Optional[float]:
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

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(movie, frames)
    self._create_degree_trackbars(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      degree = self._read_degree_trackbars(movie)
      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()
      rotated = self._get_rotated_frame(img, degree)

      if help_exists:
        h = ["[rotate]", "select degree", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("now: {0:.2f}s".format(tgt_frame / fps))
        h.append("degree: {0:.2f}".format(degree))
        self._add_texts_upper_left(rotated, h)

      cv2.imshow(movie, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0:.2f})".format(degree))
        cv2.destroyAllWindows()
        return degree

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class RotatingPicture(ABCRotatingProcess):
  """class to rotate picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, degree=degree)

  def execute(self) -> Optional[List[str]]:
    """rotating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "rotated")
    return_list: List[str] = []

    for picture, output_path in zip(self._get_target_list(), output_path_list):

      name = str(output_path / pathlib.Path(picture).name)
      img = cv2.imread(picture)

      if self._get_degree() is not None:
        degree = self._get_degree()
      else:
        degree = self.__select_degree(name, img)

      if degree is None:
        continue

      print("rotating picture '{0}'...".format(picture))
      return_list.append(name)
      rotated = self._get_rotated_frame(img, degree)
      f = rotated if self._is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
      self._create_output_directory(name)
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

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_degree_trackbars(picture)

    while True:

      degree = self._read_degree_trackbars(picture)
      rotated = self._get_rotated_frame(img, degree)

      if help_exists:
        h = ["[rotate]", "select degree", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("degree: {0:.2f}".format(degree))
        self._add_texts_upper_left(rotated, h)

      cv2.imshow(picture, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0:.2f})".format(degree))
        cv2.destroyAllWindows()
        return degree

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class RotatingPictureDirectory(ABCRotatingProcess):
  """class to rotate picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      degree: Optional[float] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored, degree=degree)

  def execute(self) -> Optional[List[str]]:
    """rotating process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "rotated")
    return_list: List[str] = []

    for directory, output_path in zip(self._get_target_list(), output_path_list):

      directory_path = pathlib.Path(directory)
      p_list = [str(p) for p in list(directory_path.iterdir())]

      if not p_list:
        print("no file exists in '{0}'!".format(directory))
        continue

      if self._get_degree() is not None:
        degree = self._get_degree()
      else:
        degree = self.__select_degree(str(output_path), p_list)

      if degree is None:
        continue

      return_list.append(str(output_path))
      print("rotating picture in '{0}'...".format(directory))

      for p in p_list:

        img = cv2.imread(p)
        f1 = self._get_rotated_frame(img, degree)
        f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        self._create_output_directory(str(output_path / pathlib.Path(p).name))
        cv2.imwrite(str(output_path / pathlib.Path(p).name), f2)

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

    print("--- rotate ---\nselect rotation degree in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists = False
    self._create_frame_trackbars(directory, len(picture_list) - 1)
    self._create_degree_trackbars(directory)

    while True:

      tgt_frame = self._read_frame_trackbars(directory, len(picture_list) - 1)
      degree = self._read_degree_trackbars(directory)
      img = cv2.imread(picture_list[tgt_frame])
      rotated = self._get_rotated_frame(img, degree)

      if help_exists:
        h = ["[rotate]", "select degree", "s:save", "h:on/off help", "q/esc:abort"]
        h.append("frame: {0}".format(tgt_frame))
        h.append("degree: {0:.2f}".format(degree))
        self._add_texts_upper_left(rotated, h)

      cv2.imshow(directory, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0:.2f})".format(degree))
        cv2.destroyAllWindows()
        return degree

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class ABCSubtitlingProcess(ABCProcess, metaclass=ABCMeta):
  """abstract base class of subtitling process"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      text: str = None,
      position: Optional[Tuple[int, int]] = None,
      size: Optional[float] = None,
      time: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        text (Optional[str], optional): text to be added into movie/picture. Defaults to None.
        position (Optional[Tuple[int, int]], optional): position of left end of text (int) [pixel]. Defaults to None. if this is not given, you will select this in GUI window.
        size (Optional[float], optional): size of text. Defaults to None. if this is not
        given, you will select this in GUI window.
        time (Optional[Tuple[float, float]]): time at beginning and end of subtitling (float) [s]. this argument is neglected for picture or directory. Defaults to None. if this is not given, you will select this in GUI window.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__text = text
    self.__position = position
    self.__size = size
    self.__time = time

  @abstractmethod
  def execute(self) -> Optional[List[str]]:
    """subtitling process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    pass

  def _get_text(self) -> Optional[str]:
    return self.__text

  def _get_position(self) -> Optional[Tuple[int, int]]:
    return self.__position

  def _get_size(self) -> Optional[float]:
    return self.__size

  def _get_time(self) -> Optional[Tuple[float, float]]:
    return self.__time

  def _create_start_trackbar(self, cv2_window: str):
    """create 'start cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("start cap\n", cv2_window, 0, 1, no)

  def _read_start_trackbar(self, cv2_window: str,
                           is_start_on: bool) -> Tuple[bool, bool]:
    """read values from 'start cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "start cap\n", is_start_on)

  def _create_stop_trackbar(self, cv2_window: str):
    """create 'stop cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("stop cap\n", cv2_window, 0, 1, no)

  def _read_stop_trackbar(self, cv2_window: str, is_stop_on: bool) -> Tuple[bool, bool]:
    """read values from 'stop cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "stop cap\n", is_stop_on)

  def _create_size_trackbars(self, cv2_window: str):
    """create 'size\n*0.1', 'size\n*1.0' trackbars for cv2 GUI"""
    cv2.createTrackbar("size\n*0.1", cv2_window, 10, 10, no)
    cv2.createTrackbar("size\n*1.0", cv2_window, 1, 10, no)

  def _read_size_trackbars(self, cv2_window: str) -> float:
    """read values from 'size\n*0.1', 'size\n*1.0' trackbars

    Returns:
        Tuple[float, float]: scaling factor
    """
    s_01 = cv2.getTrackbarPos("size\n*0.1", cv2_window)
    s_10 = cv2.getTrackbarPos("size\n*1.0", cv2_window)
    return (1 if s_01 == 0 else s_01) * 0.1 * (1 if s_10 == 0 else s_10)

  def _mouse_on_select_subtitling_position(self, event, x, y, flags, params):
    """call back function on mouse click
    """
    point = params
    if event == cv2.EVENT_LBUTTONUP:
      if len(point) >= 1:
        point.clear()
      point.append((x, y))


class SubtitlingMovie(ABCSubtitlingProcess):
  """class to subtitle movie
  """

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      text: Optional[str] = None,
      position: Optional[Tuple[int, int]] = None,
      size: Optional[float] = None,
      time: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        text (Optional[str], optional): text to be added into movie/picture. Defaults to None.
        position (Optional[Tuple[int, int]], optional): position of left end of text (int) [pixel]. Defaults to None. if this is not given, you will select this in GUI window.
        size (Optional[float], optional): size of text. Defaults to None. if this is not
        given, you will select this in GUI window.
        time (Optional[Tuple[float, float]]): time at beginning and end of subtitling (float) [s]. this argument is neglected for picture or directory. Defaults to None. if this is not given, you will select this in GUI window.
    """
    super().__init__(
        target_list=target_list,
        is_colored=is_colored,
        text=text,
        position=position,
        size=size,
        time=time,
    )

  def execute(self) -> Optional[List[str]]:
    """subtitling process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "subtitled")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      position, size, time = self.__select_subtitle_params(
          output_name,
          frames,
          fps,
          cap,
          self._get_text(),
          self._get_position(),
          self._get_size(),
          self._get_time(),
      )

    #   if degree is None:
    #     continue

    #   size_rot = self._get_rotated_frame_size(W, H, degree)
    #   center_rot = (int(size_rot[0] / 2), int(size_rot[1] / 2))
    #   center = (int(W / 2), int(H / 2))
    #   affine = self._get_rotating_affine_matrix(center, center_rot, degree)

    #   cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #   return_list.append(output_name)
    #   self._create_output_directory(output_name)
    #   fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #   output = cv2.VideoWriter(output_name, fourcc, fps, size_rot, self._is_colored())
    #   print("rotating movie '{0}'...".format(movie))

    #   while True:

    #     ret, frame = cap.read()
    #     if not ret:
    #       break
    #     f1 = cv2.warpAffine(frame, affine, size_rot, flags=cv2.INTER_CUBIC)
    #     f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    #     output.write(f2)

    # return return_list if return_list else None

  def __select_subtitle_params(
      self,
      movie: str,
      frames: int,
      fps: float,
      cap: cv2.VideoCapture,
      text: Optional[str],
      position: Optional[Tuple[int, int]],
      size: Optional[float],
      time: Optional[Tuple[float, float]],
  ):
    """select(get) subtitle parameters

    Args:
        movie (str): movie file name
        frames (int): number of total frames of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object
        text (Optional[str], optional): text to be added into movie/picture.
        position (Optional[Tuple[int, int]], optional): position of left end of text (int) [pixel].
        size (Optional[float], optional): size of text. Defaults to None.
        time (Optional[Tuple[float, float]]): time at beginning and end of subtitling (float) [s].
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)

    print("--- subtitle ---\nselect position, size, time (if necessary) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists, is_start_on, is_stop_on = False, False, False
    start_time, stop_time = 0.0, 0.0
    point: List[Tuple[int, int]] = []

    self._create_frame_trackbars(movie, frames)
    self._create_start_trackbar(movie)
    self._create_stop_trackbar(movie)
    self._create_size_trackbars(movie)

    if position:
      point.append(position)
    else:
      cv2.setMouseCallback(movie, self._mouse_on_select_subtitling_position, point)
      print("(c:clear, click:select)")

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)
      size = self._read_size_trackbars(movie)

      is_start_on, is_changed = self._read_start_trackbar(movie, is_start_on)
      if is_changed:
        start_time = tgt_frame / fps if is_start_on else 0.0

      is_stop_on, is_changed = self._read_stop_trackbar(movie, is_stop_on)
      if is_changed:
        stop_time = tgt_frame / fps if is_stop_on else 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if point:
        cv2.putText(img, text, point[0], cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                    10)
        cv2.putText(img, text, point[0], cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)

      if help_exists:
        h = ["[subtitle]", "select position, size, time", "s: save"]
        h.extend(["h: on/off help", "c: clear", "click: select", "q/esc: abort"])
        h.append("now: {0:.2f}s".format(tgt_frame / fps))
        h.append("start: {0:.2f}s".format(start_time))
        h.append("stop: {0:.2f}s".format(stop_time))
        h.append("size: {0:.2f}s".format(size))
        h.extend(["{0}".format(p) for p in point])
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        continue
        # print("'s' is pressed. threshold ({0}, {1})".format(low, high))

        # if high <= low:
        #   print("high luminance threshold must be > low")
        #   continue
        # else:
        #   cv2.destroyAllWindows()
        #   return (low, high)

      elif k == ord("c") and position is None:
        print("'c' is pressed. selected point is cleared")
        point.clear()
        continue

      elif k == ord("h"):
        help_exists = False if help_exists else True

      else:
        if self._press_q_or_Esc(k):
          return None


class SubtitlingPicture(ABCSubtitlingProcess):
  """class to subtitle picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      text: Optional[str] = None,
      position: Optional[Tuple[int, int]] = None,
      size: Optional[float] = None,
      time: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        text (Optional[str], optional): text to be added into movie/picture. Defaults to None.
        position (Optional[Tuple[int, int]], optional): position of left end of text (int) [pixel]. Defaults to None. if this is not given, you will select this in GUI window.
        size (Optional[float], optional): size of text. Defaults to None. if this is not
        given, you will select this in GUI window.
        time (Optional[Tuple[float, float]]): time at beginning and end of subtitling (float) [s]. this argument is neglected for picture or directory. Defaults to None. if this is not given, you will select this in GUI window.
    """
    super().__init__(
        target_list=target_list,
        is_colored=is_colored,
        text=text,
        position=position,
        size=size,
        time=None,
    )

  def execute(self) -> Optional[List[str]]:
    """subtitling process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "subtitled")
    return_list: List[str] = []

    # for picture, output_path in zip(self._get_target_list(), output_path_list):

    #   name = str(output_path / pathlib.Path(picture).name)
    #   img = cv2.imread(picture)

    #   if self._get_degree() is not None:
    #     degree = self._get_degree()
    #   else:
    #     degree = self.__select_degree(name, img)

    #   if degree is None:
    #     continue

    #   print("rotating picture '{0}'...".format(picture))
    #   return_list.append(name)
    #   rotated = self._get_rotated_frame(img, degree)
    #   f = rotated if self._is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    #   self._create_output_directory(name)
    #   cv2.imwrite(name, f)

    # return return_list if return_list else None


class SubtitlingPictureDirectory(ABCSubtitlingProcess):
  """class to subtitle picture"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      text: Optional[str] = None,
      position: Optional[Tuple[int, int]] = None,
      size: Optional[float] = None,
      time: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        text (Optional[str], optional): text to be added into movie/picture. Defaults to None.
        position (Optional[Tuple[int, int]], optional): position of left end of text (int) [pixel]. Defaults to None. if this is not given, you will select this in GUI window.
        size (Optional[float], optional): size of text. Defaults to None. if this is not
        given, you will select this in GUI window.
        time (Optional[Tuple[float, float]]): time at beginning and end of subtitling (float) [s]. this argument is neglected for picture or directory. Defaults to None. if this is not given, you will select this in GUI window.
    """
    super().__init__(
        target_list=target_list,
        is_colored=is_colored,
        text=text,
        position=position,
        size=size,
        time=None,
    )

  def execute(self) -> Optional[List[str]]:
    """subtitling process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "subtitled")
    # return_list: List[str] = []

    # for directory, output_path in zip(self._get_target_list(), output_path_list):

    #   directory_path = pathlib.Path(directory)
    #   p_list = [str(p) for p in list(directory_path.iterdir())]

    #   if not p_list:
    #     print("no file exists in '{0}'!".format(directory))
    #     continue

    #   if self._get_degree() is not None:
    #     degree = self._get_degree()
    #   else:
    #     degree = self.__select_degree(str(output_path), p_list)

    #   if degree is None:
    #     continue

    #   return_list.append(str(output_path))
    #   print("rotating picture in '{0}'...".format(directory))

    #   for p in p_list:

    #     img = cv2.imread(p)
    #     f1 = self._get_rotated_frame(img, degree)
    #     f2 = f1 if self._is_colored() else cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    #     self._create_output_directory(str(output_path / pathlib.Path(p).name))
    #     cv2.imwrite(str(output_path / pathlib.Path(p).name), f2)

    # return return_list if return_list else None


class TrimmingMovie(ABCProcess):
  """class to trim movie"""

  def __init__(
      self,
      *,
      target_list: List[str] = [],
      is_colored: bool = False,
      times: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        target_list (List[str], optional): list of target inputs. Defaults to [].
        is_colored (bool, optional): flag to output in color. Defaults to False.
        times (Tuple[float, float], optional): [start, stop] parameters for trimming (s). If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    super().__init__(target_list=target_list, is_colored=is_colored)
    self.__times = times

  def __get_times(self) -> Optional[Tuple[float, float]]:
    return self.__times

  def execute(self) -> Optional[List[str]]:
    """trimming process

    Returns:
        Optional[List[str]]: list of output path names. if process is not executed, None is returned.
    """
    output_path_list = self._get_output_path(self._get_target_list(), "trimmed")
    return_list: List[str] = []

    for movie, output_path in zip(self._get_target_list(), output_path_list):

      output_name = str(output_path / pathlib.Path(movie).stem) + ".mp4"
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = self._get_movie_info(cap)

      if self.__get_times() is not None:
        time = self.__get_times()
      else:
        time = self.__select_times(str(output_path), frames, fps, cap)

      if time is None:
        continue

      if time[1] <= time[0]:
        print("stop ({0}) must be > start ({1})".format(time[1], time[0]))
        continue

      if frames < round(time[1] * fps):
        print("stop frame ({0}) must be < ({1})".format(round(time[1] * fps), frames))
        continue

      current_frame, max_frame = round(time[0] * fps), round(time[1] * fps)
      cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
      return_list.append(output_name)
      self._create_output_directory(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (W, H), self._is_colored())
      print("trimming movie '{0}'...".format(movie))

      while current_frame <= max_frame:

        ret, frame = cap.read()
        if not ret:
          break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        f = frame if self._is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output.write(f)

    return return_list if return_list else None

  def __select_times(self, movie: str, frames: int, fps: float,
                     cap: cv2.VideoCapture) -> Optional[Tuple[float, float]]:
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

    print("--- trim ---\nselect time (start, stop) in GUI window!")
    print("(s: save, h:on/off help, q/esc: abort)")

    help_exists, is_start_on, is_stop_on = False, False, False
    start_time, stop_time = 0.0, 0.0
    self._create_frame_trackbars(movie, frames)
    self.__create_start_trackbar(movie)
    self.__create_stop_trackbar(movie)

    while True:

      tgt_frame = self._read_frame_trackbars(movie, frames)

      is_start_on, is_changed = self.__read_start_trackbar(movie, is_start_on)
      if is_changed:
        start_time = tgt_frame / fps if is_start_on else 0.0

      is_stop_on, is_changed = self.__read_stop_trackbar(movie, is_stop_on)
      if is_changed:
        stop_time = tgt_frame / fps if is_stop_on else 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, tgt_frame)
      ret, img = cap.read()

      if help_exists:
        h = ["[trim]", "select start,stop", "s: save", "q/esc: abort"]
        h.append("now: {0:.2f}s".format(tgt_frame / fps))
        h.append("start: {0:.2f}s".format(start_time))
        h.append("stop: {0:.2f}s".format(stop_time))
        self._add_texts_upper_left(img, h)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):

        print("'s' is pressed. start, stop ({0},{1})".format(start_time, stop_time))

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
        if self._press_q_or_Esc(k):
          return None

  def __create_start_trackbar(self, cv2_window: str):
    """create 'start cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("start cap\n", cv2_window, 0, 1, no)

  def __read_start_trackbar(self, cv2_window: str,
                            is_start_on: bool) -> Tuple[bool, bool]:
    """read values from 'start cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "start cap\n", is_start_on)

  def __create_stop_trackbar(self, cv2_window: str):
    """create 'stop cap\n' trackbar for cv2 GUI"""
    cv2.createTrackbar("stop cap\n", cv2_window, 0, 1, no)

  def __read_stop_trackbar(self, cv2_window: str,
                           is_stop_on: bool) -> Tuple[bool, bool]:
    """read values from 'stop cap\n' trackbar

    Returns:
        Tuple[bool, bool]: [is_trackbar_on, is_trackbar_changed]
    """
    return self._read_bool_trackbar(cv2_window, "stop cap\n", is_stop_on)
