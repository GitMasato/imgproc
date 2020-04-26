"""process module containing image process functions
"""
import cv2
import numpy
import pathlib
import shutil
from matplotlib import pyplot
from typing import List, Optional, Protocol, Tuple


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
  path_list: List[pathlib.Path] = []

  for target in target_list:
    target_path = pathlib.Path(target).resolve()

    if not target_path.is_dir() and not target_path.is_file():
      print("'{0}' does not exist!".format(str(target_path)))
      continue

    layers = target_path.parts
    if "cv2" in layers:
      p_path = target_path.parents[1] if target_path.is_file() else target_path.parent
      path_list.append(pathlib.Path(p_path / output))
    else:
      p = target_path.stem if target_path.is_file() else target_path.name
      path_list.append(pathlib.Path(pathlib.Path.cwd() / "cv2" / p / output))

  return path_list


def clean_output_directory(output_path_list: List[pathlib.Path]):
  """clean output directory

  Args:
      output_path_list (List[pathlib.Path]): list of output directories (pathlib.Path objects)
  """
  output_path_group = set(output_path_list)

  for output_path in output_path_group:
    if not output_path.is_dir():
      output_path.mkdir(parents=True)
    else:
      if list(output_path.iterdir()):
        shutil.rmtree(output_path)
        output_path.mkdir()


def no(no):
  """meaningless function just for trackbar callback"""
  pass


class ABCProcess(Protocol):
  """abstract base class for image processing class"""

  def execute(self):
    pass


class AnimatingPicture:
  """class to create animation (movie) from pictures"""

  def __init__(
    self,
    *,
    picture_list: Optional[List[str]] = None,
    is_colored: bool = False,
    fps: Optional[float] = None,
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures, directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): whether to output in color. Defaults to False.
        fps (Optional[float], optional): fps of created movie. Defaults to None.
    """
    self.__picture_list = picture_list
    self.__is_colored = is_colored
    self.__fps = fps

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_fps(self) -> Optional[float]:
    return self.__fps

  def execute(self):
    """animating process to create movie

    Returns:
        Tuple[str, List[str]]: type of output, list of output (movie) path names
    """
    if self.get_fps() is not None:
      fps = self.get_fps()

    output_path_list = get_output_path(self.get_picture_list(), "animated")
    clean_output_directory(output_path_list)
    is_first_unprocessed = True
    unprocessed_pictures: List[pathlib.Path] = []
    return_list: List[str] = []

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        unprocessed_pictures.append(picture_path)

        if is_first_unprocessed is True:
          img = cv2.imread(str(picture_path))
          unprocessed_size = (int(img.shape[1]), int(img.shape[0]))
          unprocessed_name = str(pathlib.Path(output_path / picture_path.stem)) + ".mp4"
          return_list.append(unprocessed_name)
          is_first_unprocessed = False

      elif picture_path.is_dir():

        print("animating pictures in '{0}'...".format(picture))
        p_list = list(picture_path.iterdir())
        if self.get_fps() is None:
          fps = self.select_fps(picture_path.name, p_list)
          if fps is None:
            continue

        img = cv2.imread(str(p_list[0]))
        size = (int(img.shape[1]), int(img.shape[0]))
        output_name = str(pathlib.Path(output_path / picture_path.name)) + ".mp4"
        return_list.append(output_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output = cv2.VideoWriter(output_name, fourcc, fps, size, self.is_colored())

        for p in p_list:
          img = cv2.imread(str(p))
          output.write(
            img if self.is_colored() else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          )

    if unprocessed_pictures:

      print("animating rest of pictures...")
      if self.get_fps() is None:
        fps = self.select_fps(unprocessed_name, unprocessed_pictures)
        if fps:
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          unprocessed_output = cv2.VideoWriter(
            unprocessed_name, fourcc, fps, unprocessed_size, self.is_colored()
          )

          for p in unprocessed_pictures:
            img = cv2.imread(str(p))
            unprocessed_output.write(
              img if self.is_colored() else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            )

    return ("movie", return_list) if return_list else (None, [])

  def select_fps(
    self, movie_name: str, picture_path_list: List[pathlib.Path]
  ) -> Optional[float]:
    """select(get) fps for animating using GUI window

    Args:
        movie_name (str): movie name
        picture_path_list (List[pathlib.Path]): picture path list

    Returns:
        Optional[float]: fps of movie
    """
    cv2.namedWindow(movie_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie_name, 500, 700)
    help_exists = True

    frames = len(picture_path_list) - 1
    division = 50
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", movie_name, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie_name, 0, tick_s, no)
    cv2.createTrackbar("fps\n", movie_name, 10, 50, no)

    print("--- animate ---")
    print("select fps in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie_name) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie_name)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      fps_read = cv2.getTrackbarPos("fps\n", movie_name)
      fps = 1 if fps_read == 0 else fps_read
      img = cv2.imread(str(picture_path_list[frame_now]))

      if help_exists:
        add_texts_upper_left(
          img,
          [
            "[animate]",
            "select fps",
            "now: {0}".format(frame_now),
            "fps: {0}".format(fps),
          ],
        )
        add_texts_lower_right(
          img, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(movie_name, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. fps is saved ({0})".format(fps))
        cv2.destroyAllWindows()
        return fps

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class BinarizingMovie:
  """class to binarize movie
  """

  def __init__(
    self,
    *,
    movie_list: Optional[List[str]] = None,
    thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        movie_list (Optional[List[str]], optional): list of movies. Defaults to None.
        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__movie_list = movie_list
    self.__thresholds = thresholds

  def get_movie_list(self) -> Optional[List[str]]:
    return self.__movie_list

  def get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def execute(self):
    """binarizing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (movie) path names
    """
    if self.get_thresholds() is not None:
      thresholds = self.get_thresholds()
      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        return (None, [])

    output_path_list = get_output_path(self.get_movie_list(), "binarized")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for movie, output_path in zip(self.get_movie_list(), output_path_list):

      print("binarizing movie '{0}'...".format(movie))
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      if self.get_thresholds() is None:
        thresholds = self.select_thresholds(movie, frames, fps, cap)
        if thresholds is None:
          continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, (int(W), int(H)), False)

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
        ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
        output.write(bin_2)

    return ("movie", return_list) if return_list else (None, [])

  def select_thresholds(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        movie (str): movie file name
        frames (int): total frame of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = True

    division = 100
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", movie, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie, 0, tick_s, no)
    cv2.createTrackbar("low\n", movie, 0, 255, no)
    cv2.createTrackbar("high\n", movie, 255, 255, no)

    print("--- binarize ---")
    print("select threshold in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      low = cv2.getTrackbarPos("low\n", movie)
      high = cv2.getTrackbarPos("high\n", movie)

      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
      ret, img = cap.read()
      ret, bin_1 = cv2.threshold(img, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(
          bin_2,
          [
            "[binarize]",
            "select thresholds",
            "now: {0:.2f}s".format(frame_now / fps),
            "low: {0}".format(low),
            "high: {0}".format(high),
          ],
        )
        add_texts_lower_right(
          bin_2, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )
        if high <= low:
          add_texts_lower_left(bin_2, ["high must be > low"])

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
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class BinarizingPicture:
  """class to binarize picture
  """

  def __init__(
    self,
    *,
    picture_list: Optional[List[str]] = None,
    thresholds: Optional[Tuple[int, int]] = None,
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures, directories where pictures are stored. Defaults to None.
        thresholds (Optional[Tuple[int, int]], optional): threshold values to be used to binarize movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__picture_list = picture_list
    self.__thresholds = thresholds

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def get_thresholds(self) -> Optional[Tuple[int, int]]:
    return self.__thresholds

  def execute(self):
    """binarizing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (picture or directory) path names
    """
    if self.get_thresholds() is not None:
      thresholds = self.get_thresholds()
      if thresholds[1] <= thresholds[0]:
        print("high luminance threshold must be > low")
        return (None, [])

    output_path_list = get_output_path(self.get_picture_list(), "binarized")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        print("binarizing picture '{0}'...".format(picture))
        img = cv2.imread(picture)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.get_thresholds() is None:
          thresholds = self.select_thresholds_picture(picture, gray)
          if thresholds is None:
            continue

        ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
        ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
        name = str(pathlib.Path(output_path / picture_path.name))
        return_list.append(name)
        cv2.imwrite(name, bin_2)

      elif picture_path.is_dir():

        print("binarizing picture in '{0}'...".format(picture))
        p_list = list(picture_path.iterdir())
        if self.get_thresholds() is None:
          thresholds = self.select_thresholds_directory(picture_path.name, p_list)
          if thresholds is None:
            continue

        return_list.append(str(output_path))

        for p in p_list:
          img = cv2.imread(str(p))
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          ret, bin_1 = cv2.threshold(gray, thresholds[0], 255, cv2.THRESH_TOZERO)
          ret, bin_2 = cv2.threshold(bin_1, thresholds[1], 255, cv2.THRESH_TOZERO_INV)
          cv2.imwrite(str(pathlib.Path(output_path / p.name)), bin_2)

    return ("picture", return_list) if return_list else (None, [])

  def select_thresholds_picture(
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
    help_exists = True

    cv2.createTrackbar("low\n", picture, 0, 255, no)
    cv2.createTrackbar("high\n", picture, 255, 255, no)
    print("--- binarize ---")
    print("select threshold in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      low = cv2.getTrackbarPos("low\n", picture)
      high = cv2.getTrackbarPos("high\n", picture)
      ret, bin_1 = cv2.threshold(img, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(bin_2, ["binarize:", "select threshold"])
        add_texts_lower_right(
          bin_2, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )
        if high <= low:
          add_texts_lower_left(bin_2, ["high must be > low"])

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
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def select_thresholds_directory(
    self, directory: str, picture_path_list: List[pathlib.Path]
  ) -> Optional[Tuple[int, int]]:
    """select(get) threshold values for binarization using GUI window

    Args:
        directory (str): directory name
        picture_path_list (List[pathlib.Path]): picture path list

    Returns:
        Optional[Tuple[int, int]]: [low, high] threshold values
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = True

    frames = len(picture_path_list) - 1
    division = 50
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", directory, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", directory, 0, tick_s, no)
    cv2.createTrackbar("low\n", directory, 0, 255, no)
    cv2.createTrackbar("high\n", directory, 255, 255, no)

    print("--- binarize ---")
    print("select threshold in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", directory) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", directory)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      low = cv2.getTrackbarPos("low\n", directory)
      high = cv2.getTrackbarPos("high\n", directory)

      img = cv2.imread(str(picture_path_list[frame_now]))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, bin_1 = cv2.threshold(gray, low, 255, cv2.THRESH_TOZERO)
      ret, bin_2 = cv2.threshold(bin_1, high, 255, cv2.THRESH_TOZERO_INV)

      if help_exists:
        add_texts_upper_left(
          bin_2,
          [
            "[binarize]",
            "select thresholds",
            "now: {0}".format(frame_now),
            "low: {0}".format(low),
            "high: {0}".format(high),
          ],
        )
        add_texts_lower_right(
          bin_2, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )
        if high <= low:
          add_texts_lower_left(bin_2, ["high must be > low"])

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
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class CapturingMovie:
  """class to capture movie"""

  def __init__(
    self,
    *,
    movie_list: Optional[List[str]] = None,
    is_colored: bool = False,
    times: Optional[Tuple[float, float, float]] = None,
  ):
    """constructor

    Args:
        movie_list (List[str], optional): list of movies. Defaults to None.
        is_colored (bool, optional):  flag to output in color. Defaults to False.
        times (Tuple[float, float, float], optional): [start, stop, step] parameters for capturing movie (s). If this variable is None, this will be selected using GUI window Defaults to None.
    """
    self.__movie_list = movie_list
    self.__is_colored = is_colored
    self.__times = times

  def get_movie_list(self) -> Optional[List[str]]:
    return self.__movie_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_times(self) -> Optional[Tuple[float, float, float]]:
    return self.__times

  def execute(self):
    """capturing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (directory) path names
    """
    if self.get_times() is not None:
      time = self.get_times()

      if time[1] - time[0] <= time[2]:
        print("difference between stop and start must be > time step")
        return (None, [])
      if time[1] <= time[0]:
        print("stop must be > start")
        return (None, [])
      if time[2] < 0.001:
        print("time step must be > 0")
        return (None, [])

    output_path_list = get_output_path(self.get_movie_list(), "captured")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for movie, output_path in zip(self.get_movie_list(), output_path_list):

      print("capturing movie '{0}'...".format(movie))
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      if self.get_times() is None:
        time = self.select_times(movie, frames, fps, cap)
        if time is None:
          continue

      capture_time = time[0]
      return_list.append(str(output_path))

      while capture_time <= time[1]:

        cap.set(cv2.CAP_PROP_POS_FRAMES, round(capture_time * fps))
        ret, frame = cap.read()
        cv2.imwrite(
          "{0}/{1:08}_ms.png".format(
            str(output_path), int(round(capture_time - time[0], 3) * 1000)
          ),
          frame if self.is_colored() else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        )
        capture_time += time[2]

    return ("picture", return_list) if return_list else (None, [])

  def select_times(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[float, float, float]]:
    """select(get) parametes for capture using GUI window

    Args:
        movie (str): movie file name
        frames (int): total frame of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[float, float, float]]: [start, stop, step] parameters for capturing movie (s).
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = True

    division = 100
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    is_start_on, is_stop_on = False, False
    start_time, stop_time = 0.0, 0.0
    warning_message: List[str] = []

    cv2.createTrackbar("frame\n", movie, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie, 0, tick_s, no)
    cv2.createTrackbar("start cap\n", movie, 0, 1, no)
    cv2.createTrackbar("stop cap\n", movie, 0, 1, no)
    cv2.createTrackbar("step 10ms\n", movie, 1, division, no)
    print("--- capture ---")
    print("select time parameters in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      time_step = cv2.getTrackbarPos("step 10ms\n", movie) * 10 * 0.001

      if not is_start_on:
        if cv2.getTrackbarPos("start cap\n", movie):
          is_start_on, start_time = True, frame_now / fps
      else:
        if not cv2.getTrackbarPos("start cap\n", movie):
          is_start_on, start_time = False, 0.0
      if not is_stop_on:
        if cv2.getTrackbarPos("stop cap\n", movie):
          is_stop_on, stop_time = True, frame_now / fps
      else:
        if not cv2.getTrackbarPos("stop cap\n", movie):
          is_stop_on, stop_time = False, 0.0

      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
      ret, img = cap.read()

      if help_exists:
        add_texts_upper_left(
          img,
          [
            "[capture]",
            "select start,stop,step",
            "now: {0:.2f}s".format(frame_now / fps),
            "start: {0:.2f}s".format(start_time),
            "stop: {0:.2f}s".format(stop_time),
            "step: {0:.2f}s".format(time_step),
          ],
        )
        add_texts_lower_right(img, ["s: save", "q/esc: abort"])
        add_texts_lower_left(img, warning_message)

      cv2.imshow(movie, img)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        if stop_time - start_time <= time_step:
          warning_message = ["stop-start must be > step"]
          continue
        if stop_time <= start_time:
          warning_message = ["stop must be > start"]
          continue
        if time_step < 0.001:
          warning_message = ["step must be > 0"]
          continue
        print("'s' is pressed. capture parameters are saved")
        cv2.destroyAllWindows()
        return (start_time, stop_time, time_step)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esc' is pressed. abort")
        return None


class CroppingMovie:
  """class to crop movie"""

  def __init__(
    self,
    *,
    movie_list: Optional[List[str]] = None,
    is_colored: bool = False,
    positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        movie_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__movie_list = movie_list
    self.__is_colored = is_colored
    self.__positions = positions

  def get_movie_list(self) -> Optional[List[str]]:
    return self.__movie_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def execute(self):
    """capturing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (movie) path names
    """
    if self.get_positions() is not None:
      positions = self.get_positions()
      if positions[2] <= positions[0] or positions[3] <= positions[1]:
        print("2nd position must be > 1st")
        return (None, [])

    output_path_list = get_output_path(self.get_movie_list(), "cropped")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for movie, output_path in zip(self.get_movie_list(), output_path_list):

      print("cropping movie '{0}'...".format(movie))
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      if self.get_positions() is None:
        positions = self.select_positions(movie, W, H, frames, fps, cap)
        if positions is None:
          continue

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      size = (int(positions[2] - positions[0]), int(positions[3] - positions[1]))
      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self.is_colored())

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        cropped = frame[positions[1] : positions[3], positions[0] : positions[2]]
        output.write(
          cropped if self.is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        )

    return ("movie", return_list) if return_list else (None, [])

  def select_positions(
    self, movie: str, W: int, H: int, frames: int, fps: float, cap: cv2.VideoCapture,
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for capring process using GUI window

    Args:
        movie (str): movie file name
        W (int): W video length
        H (int): H video length
        frames (int): total frame of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = True

    division = 100
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", movie, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie, 0, tick_s, no)

    points: List[Tuple[int, int]] = []
    warning_message: List[str] = []
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(movie, self.mouse_on_select_positions, points)
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select positions in GUI window!")
    print("(s: save if selected, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
      ret, img = cap.read()

      if len(points) == 1:
        cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
          warning_message = ["2nd must be > 1st"]
        else:
          cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()
        warning_message = ["3rd is not accepted"]

      if help_exists:
        add_texts_lower_right(
          img,
          [
            "s:save if selected",
            "h:on/off help",
            "c:clear",
            "click:select",
            "q/esc:abort",
          ],
        )
        add_texts_upper_left(
          img,
          ["[crop]", "select two positions", "now: {0:.2f}s".format(frame_now / fps)],
        )
        add_texts_lower_left(img, warning_message)
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
          warning_message = ["not selected yet"]
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        warning_message = ["cleared"]
        continue

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def mouse_on_select_positions(self, event, x, y, flags, params):
    """call back function on mouse click
    """
    points = params

    if event == cv2.EVENT_LBUTTONUP:
      points.append([x, y])


class CroppingPicture:
  """class to capture picture"""

  def __init__(
    self,
    *,
    picture_list: Optional[List[str]] = None,
    is_colored: bool = False,
    positions: Optional[Tuple[int, int, int, int]] = None,
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures, directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        positions (Optional[Tuple[int, int, int, int]], optional): [x_1, y_1,x_2, y_2] two positions to crop movie. If this variable is None, this will be selected using GUI window. Defaults to None.
    """
    self.__picture_list = picture_list
    self.__is_colored = is_colored
    self.__positions = positions

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_positions(self) -> Optional[Tuple[int, int, int, int]]:
    return self.__positions

  def execute(self):
    """cropping process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (picture or directory) path names
    """
    if self.get_positions() is not None:
      positions = self.get_positions()
      if positions[2] <= positions[0] or positions[3] <= positions[1]:
        print("2nd position must be larger than 1st")
        return ("picture", [])

    output_path_list = get_output_path(self.get_picture_list(), "cropped")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        print("cropping picture '{0}'...".format(picture))
        img = cv2.imread(picture)
        if self.get_positions() is None:
          positions = self.select_positions_picture(picture, img)
          if positions is None:
            continue

        cropped = img[positions[1] : positions[3], positions[0] : positions[2]]
        name = str(pathlib.Path(output_path / picture_path.name))
        return_list.append(name)
        cv2.imwrite(
          name,
          cropped if self.is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
        )

      elif picture_path.is_dir():

        print("cropping picture in '{0}'...".format(picture))
        p_list = list(picture_path.iterdir())
        if self.get_positions() is None:
          positions = self.select_positions_directory(picture_path.name, p_list)
          if positions is None:
            continue

        return_list.append(str(output_path))

        for p in p_list:
          img = cv2.imread(str(p))
          cropped = img[positions[1] : positions[3], positions[0] : positions[2]]
          cv2.imwrite(
            str(pathlib.Path(output_path / p.name)),
            cropped if self.is_colored() else cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
          )

    return ("picture", return_list) if return_list else (None, [])

  def select_positions_picture(
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
    warning_message: List[str] = []

    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(picture, self.mouse_on_select_positions, points)
    help_exists = True
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select positions in GUI window!")
    print("(s: save if selected, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      img_show = img.copy()

      if len(points) == 1:
        cv2.line(img_show, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img_show, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
          warning_message = ["2nd must be > 1st"]
        else:
          cv2.line(img_show, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img_show, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img_show, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img_show, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()
        warning_message = ["3rd is not accepted"]

      if help_exists:
        add_texts_lower_right(
          img_show,
          [
            "s:save if selected",
            "h:on/off help",
            "c:clear",
            "click:select",
            "q/esc:abort",
          ],
        )
        add_texts_upper_left(
          img_show, ["[crop]", "select two positions"],
        )
        add_texts_lower_left(img_show, warning_message)
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
          warning_message = ["not selected yet"]
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        warning_message = ["cleared"]
        continue

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def select_positions_directory(
    self, directory: str, picture_path_list: List[pathlib.Path]
  ) -> Optional[Tuple[int, int, int, int]]:
    """select(get) two positions for capring process using GUI window

    Args:
        directory (str): directory name
        picture_path_list (List[pathlib.Path]): picture path list

    Returns:
        Optional[Tuple[int, int, int, int]]: [x_1, y_1,x_2, y_2] two positions to crop image
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = True

    frames = len(picture_path_list) - 1
    division = 50
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", directory, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", directory, 0, tick_s, no)

    points: List[Tuple[int, int]] = []
    warning_message: List[str] = []
    cv2.setMouseCallback(directory, self.mouse_on_select_positions, points)
    line_color = (255, 255, 255)

    print("--- crop ---")
    print("select positions in GUI window!")
    print("(s: save if selected, h:on/off help, c: clear, click: select, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", directory) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", directory)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      img = cv2.imread(str(picture_path_list[frame_now]))
      W, H = img.shape[1], img.shape[0]

      if len(points) == 1:
        cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
        cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
      elif len(points) == 2:
        if points[1][1] <= points[0][1] or points[1][0] <= points[0][0]:
          points.clear()
          warning_message = ["2nd must be > 1st"]
        else:
          cv2.line(img, (points[0][0], 0), (points[0][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[0][1]), (W - 1, points[0][1]), line_color, 2)
          cv2.line(img, (points[1][0], 0), (points[1][0], H - 1), line_color, 2)
          cv2.line(img, (0, points[1][1]), (W - 1, points[1][1]), line_color, 2)
      elif len(points) == 3:
        points.clear()
        warning_message = ["3rd is not accepted"]

      if help_exists:
        add_texts_lower_right(
          img,
          [
            "s:save if selected",
            "h:on/off help",
            "c:clear",
            "click:select",
            "q/esc:abort",
          ],
        )
        add_texts_upper_left(
          img, ["[crop]", "select two positions", "frame: {0}".format(frame_now)],
        )
        add_texts_lower_left(img, warning_message)
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
          warning_message = ["not selected yet"]
          continue

      elif k == ord("c"):
        print("'c' is pressed. selected points are cleared")
        points.clear()
        warning_message = ["cleared"]
        continue

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def mouse_on_select_positions(self, event, x, y, flags, params):
    """call back function on mouse click
    """
    points = params

    if event == cv2.EVENT_LBUTTONUP:
      points.append([x, y])


class CreatingLuminanceHistgramPicture:
  """class to create luminance histgram of picture"""

  def __init__(
    self, *, picture_list: Optional[List[str]] = None, is_colored: bool = False
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures, directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
    """
    self.__picture_list = picture_list
    self.__is_colored = is_colored

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def execute(self):
    """creating luminance histglam

    Returns:
        Tuple[None, List[]]: None, []
    """
    output_path_list = get_output_path(self.get_picture_list(), "histgram_luminance")
    clean_output_directory(output_path_list)

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        print("creating luminance histgram of picture '{0}'...".format(picture))
        if self.is_colored():
          self.create_color_figure(picture_path, output_path)
        else:
          self.create_gray_figure(picture_path, output_path)

      elif picture_path.is_dir():

        print("creating luminance histgram of picture in '{0}'...".format(picture))
        if self.is_colored():
          for p in list(picture_path.iterdir()):
            self.create_color_figure(p, output_path)
        else:
          for p in list(picture_path.iterdir()):
            self.create_gray_figure(p, output_path)

    return (None, [])

  def create_color_figure(self, picture_path: pathlib.Path, output_path: pathlib.Path):
    """create output figure in color

    Args:
        picture_path (pathlib.Path): picture (pathlib.Path) object
        output_path (pathlib.Path): output directory (pathlib.Path) object
    """
    img = cv2.imread(str(picture_path))
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
    fig.savefig(str(pathlib.Path(output_path / picture_path.name)))
    pyplot.close(fig)

  def create_gray_figure(self, picture_path: pathlib.Path, output_path: pathlib.Path):
    """create output figure in gray

    Args:
        picture_path (pathlib.Path): picture (pathlib.Path) object
        output_path (pathlib.Path): output directory (pathlib.Path) object
    """
    img = cv2.imread(str(picture_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig = pyplot.figure()
    subplot = fig.add_subplot(1, 2, 1)
    subplot.imshow(gray, cmap="gray")
    subplot.axis("off")
    subplot = fig.add_subplot(1, 2, 2)
    subplot.hist(gray.flatten(), bins=numpy.arange(256 + 1))
    fig.savefig(str(pathlib.Path(output_path / picture_path.name)))
    pyplot.close(fig)


class ResizingMovie:
  """class to resize movie"""

  def __init__(
    self,
    *,
    movie_list: Optional[List[str]] = None,
    is_colored: bool = False,
    scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        movie_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale movie. Defaults to None.
    """
    self.__movie_list = movie_list
    self.__is_colored = is_colored
    self.__scales = scales

  def get_movie_list(self) -> Optional[List[str]]:
    return self.__movie_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def execute(self):
    """resizing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (movie) path names
    """
    if self.get_scales() is not None:
      scales = self.get_scales()

    output_path_list = get_output_path(self.get_movie_list(), "resized")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for movie, output_path in zip(self.get_movie_list(), output_path_list):

      print("resizing movie '{0}'...".format(movie))
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      if self.get_scales() is None:
        scales = self.select_scales(movie, W, H, frames, fps, cap)
        if scales is None:
          continue

      size = (int(W * scales[0]), int(H * scales[1]))
      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size, self.is_colored())

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        resized = cv2.resize(frame, dsize=size)
        output.write(
          resized if self.is_colored() else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        )

    return ("movie", return_list) if return_list else (None, [])

  def select_scales(
    self, movie: str, W: int, H: int, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[Tuple[float, float]]:
    """select(get) rotation degree using GUI window

    Args:
        movie (str): movie file name
        W (int): W length of movie
        H (int): H length of movie
        frames (int): total frame of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[Tuple[float, float]]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = True

    division = 100
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", movie, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie, 0, tick_s, no)

    cv2.createTrackbar("scale x\n*0.1", movie, 10, 10, no)
    cv2.createTrackbar("scale x\n*1.0", movie, 1, 10, no)
    cv2.createTrackbar("scale y\n*0.1", movie, 10, 10, no)
    cv2.createTrackbar("scale y\n*1.0", movie, 1, 10, no)

    print("--- resize ---")
    print("select scales in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie)
      frame_now = frame + frame_s if frame + frame_s < frames else frames

      s_x_01 = cv2.getTrackbarPos("scale x\n*0.1", movie)
      s_x_10 = cv2.getTrackbarPos("scale x\n*1.0", movie)
      s_y_01 = cv2.getTrackbarPos("scale y\n*0.1", movie)
      s_y_10 = cv2.getTrackbarPos("scale y\n*1.0", movie)
      s_x = (1 if s_x_01 == 0 else s_x_01) * 0.1 * (1 if s_x_10 == 0 else s_x_10)
      s_y = (1 if s_y_01 == 0 else s_y_01) * 0.1 * (1 if s_y_10 == 0 else s_y_10)
      size = (int(W * s_x), int(H * s_y))

      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
      ret, img = cap.read()
      resized = cv2.resize(img, dsize=size)

      if help_exists:
        add_texts_upper_left(
          resized,
          [
            "[resize]",
            "select scales",
            "now: {0:.2f}s".format(frame_now / fps),
            "scale: {0:.1f},{1:.1f}".format(s_x, s_y),
          ],
        )
        add_texts_lower_right(
          resized, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(movie, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales is saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class ResizingPicture:
  """class to resize picture"""

  def __init__(
    self,
    *,
    picture_list: Optional[List[str]] = None,
    is_colored: bool = False,
    scales: Optional[Tuple[float, float]] = None,
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures. directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): flag to output in color. Defaults to False.
        scales (Optional[Tuple[float, float]], optional): [x, y] ratios to scale movie. Defaults to None.
    """
    self.__picture_list = picture_list
    self.__is_colored = is_colored
    self.__scales = scales

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_scales(self) -> Optional[Tuple[float, float]]:
    return self.__scales

  def execute(self):
    """resizing process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (picture or directory) path names
    """
    if self.get_scales() is not None:
      scales = self.get_scales()

    output_path_list = get_output_path(self.get_picture_list(), "resized")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        print("resizing picture '{0}'...".format(picture))
        img = cv2.imread(picture)
        W, H = img.shape[1], img.shape[0]
        if self.get_scales() is None:
          scales = self.select_scales_picture(picture, img, W, H)
          if scales is None:
            continue

        resized = cv2.resize(img, dsize=(int(W * scales[0]), int(H * scales[1])))
        name = str(pathlib.Path(output_path / picture_path.name))
        return_list.append(name)
        cv2.imwrite(
          name,
          resized if self.is_colored() else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
        )

      elif picture_path.is_dir():

        print("resizing picture in '{0}'...".format(picture))
        p_list = list(picture_path.iterdir())
        if self.get_scales() is None:
          scales = self.select_scales_directory(picture_path.name, p_list)
          if scales is None:
            continue

        return_list.append(str(output_path))

        for p in p_list:
          img = cv2.imread(str(p))
          W, H = img.shape[1], img.shape[0]
          resized = cv2.resize(img, dsize=(int(W * scales[0]), int(H * scales[1])))
          cv2.imwrite(
            str(pathlib.Path(output_path / p.name)),
            resized if self.is_colored() else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
          )

    return ("picture", return_list) if return_list else (None, [])

  def select_scales_picture(
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
    help_exists = True

    cv2.createTrackbar("scale x\n*0.1", picture, 10, 10, no)
    cv2.createTrackbar("scale x\n*1.0", picture, 1, 10, no)
    cv2.createTrackbar("scale y\n*0.1", picture, 10, 10, no)
    cv2.createTrackbar("scale y\n*1.0", picture, 1, 10, no)

    print("--- resize ---")
    print("select scales in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      s_x_01 = cv2.getTrackbarPos("scale x\n*0.1", picture)
      s_x_10 = cv2.getTrackbarPos("scale x\n*1.0", picture)
      s_y_01 = cv2.getTrackbarPos("scale y\n*0.1", picture)
      s_y_10 = cv2.getTrackbarPos("scale y\n*1.0", picture)
      s_x = (1 if s_x_01 == 0 else s_x_01) * 0.1 * (1 if s_x_10 == 0 else s_x_10)
      s_y = (1 if s_y_01 == 0 else s_y_01) * 0.1 * (1 if s_y_10 == 0 else s_y_10)
      size = (int(W * s_x), int(H * s_y))
      resized = cv2.resize(img, dsize=size)

      if help_exists:
        add_texts_upper_left(
          resized,
          ["[resize]", "select scales", "scale: {0:.1f},{1:.1f}".format(s_x, s_y)],
        )
        add_texts_lower_right(
          resized, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(picture, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales is saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def select_scales_directory(
    self, directory: str, picture_path_list: List[pathlib.Path]
  ) -> Optional[Tuple[float, float]]:
    """select(get) resizing scales using GUI window

    Args:
        directory (str): directory name
        picture_path_list (List[pathlib.Path]): picture path list

    Returns:
        Optional[float]: [x, y] ratios to scale movie
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = True

    frames = len(picture_path_list) - 1
    division = 50
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", directory, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", directory, 0, tick_s, no)

    cv2.createTrackbar("scale x\n*0.1", directory, 10, 10, no)
    cv2.createTrackbar("scale x\n*1.0", directory, 1, 10, no)
    cv2.createTrackbar("scale y\n*0.1", directory, 10, 10, no)
    cv2.createTrackbar("scale y\n*1.0", directory, 1, 10, no)

    print("--- resize ---")
    print("select scales in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", directory) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", directory)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      img = cv2.imread(str(picture_path_list[frame_now]))
      W, H = img.shape[1], img.shape[0]

      s_x_01 = cv2.getTrackbarPos("scale x\n*0.1", directory)
      s_x_10 = cv2.getTrackbarPos("scale x\n*1.0", directory)
      s_y_01 = cv2.getTrackbarPos("scale y\n*0.1", directory)
      s_y_10 = cv2.getTrackbarPos("scale y\n*1.0", directory)
      s_x = (1 if s_x_01 == 0 else s_x_01) * 0.1 * (1 if s_x_10 == 0 else s_x_10)
      s_y = (1 if s_y_01 == 0 else s_y_01) * 0.1 * (1 if s_y_10 == 0 else s_y_10)
      size = (int(W * s_x), int(H * s_y))
      resized = cv2.resize(img, dsize=size)

      if help_exists:
        add_texts_upper_left(
          resized,
          [
            "[resize]",
            "select scales",
            "frame: {0}".format(frame_now),
            "scale: {0:.1f},{1:.1f}".format(s_x, s_y),
          ],
        )
        add_texts_lower_right(
          resized, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(directory, resized)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. scales is saved ({0:.1f},{1:.1f})".format(s_x, s_y))
        cv2.destroyAllWindows()
        return (s_x, s_y)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class RotatingMovie:
  """class to rotate movie
  """

  def __init__(
    self,
    *,
    movie_list: Optional[List[str]] = None,
    is_colored: bool = False,
    degree: Optional[float] = None,
  ):
    """constructor

    Args:
        movie_list (Optional[List[str]], optional): list of movies. Defaults to None.
        is_colored (bool, optional): [description]. flag to output in color to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    self.__movie_list = movie_list
    self.__is_colored = is_colored
    self.__degree = degree

  def get_movie_list(self) -> Optional[List[str]]:
    return self.__movie_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_degree(self) -> Optional[float]:
    return self.__degree

  def execute(self):
    """rotating process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (movie) path names
    """
    if self.get_degree() is not None:
      degree = self.get_degree()
      rad = degree / 180.0 * numpy.pi
      sin_rad = numpy.absolute(numpy.sin(rad))
      cos_rad = numpy.absolute(numpy.cos(rad))

    output_path_list = get_output_path(self.get_movie_list(), "rotated")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for movie, output_path in zip(self.get_movie_list(), output_path_list):

      print("rotating movie '{0}'...".format(movie))
      cap = cv2.VideoCapture(movie)
      W, H, frames, fps = get_movie_info(cap)
      if self.get_degree() is None:
        degree = self.select_degree(movie, frames, fps, cap)
        if degree is None:
          continue
        rad = degree / 180.0 * numpy.pi
        sin_rad = numpy.absolute(numpy.sin(rad))
        cos_rad = numpy.absolute(numpy.cos(rad))

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      center = (W / 2, H / 2)
      W_rot = int(numpy.round(H * sin_rad + W * cos_rad))
      H_rot = int(numpy.round(H * cos_rad + W * sin_rad))
      size_rot = (W_rot, H_rot)
      rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
      affine_matrix = rotation_matrix.copy()
      affine_matrix[0][2] = affine_matrix[0][2] - W / 2 + W_rot / 2
      affine_matrix[1][2] = affine_matrix[1][2] - H / 2 + H_rot / 2

      output_name = str(pathlib.Path(output_path / pathlib.Path(movie).stem)) + ".mp4"
      return_list.append(output_name)
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      output = cv2.VideoWriter(output_name, fourcc, fps, size_rot, self.is_colored())

      while True:
        ret, frame = cap.read()
        if not ret:
          break
        rotated = cv2.warpAffine(frame, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
        output.write(
          rotated if self.is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        )

    return ("movie", return_list) if return_list else (None, [])

  def select_degree(
    self, movie: str, frames: int, fps: float, cap: cv2.VideoCapture
  ) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        movie (str): movie file name
        frames (int): total frame of movie
        fps (float): fps of movie
        cap (cv2.VideoCapture): cv2 video object

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(movie, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(movie, 500, 700)
    help_exists = True

    division = 100
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", movie, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", movie, 0, tick_s, no)

    rotation_tick = 4
    rotation_tick_s = 90
    cv2.createTrackbar("degree\n", movie, 0, rotation_tick, no)
    cv2.createTrackbar("degree s\n", movie, 0, rotation_tick_s, no)

    print("--- rotate ---")
    print("select rotation degree in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", movie) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", movie)
      frame_now = frame + frame_s if frame + frame_s < frames else frames

      degree = cv2.getTrackbarPos("degree\n", movie) * rotation_tick_s
      degree_s = cv2.getTrackbarPos("degree s\n", movie)
      degree_total = degree + degree_s
      rad = degree_total / 180.0 * numpy.pi
      sin_rad = numpy.absolute(numpy.sin(rad))
      cos_rad = numpy.absolute(numpy.cos(rad))

      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
      ret, img = cap.read()
      rotated = get_rotated_image(img, degree_total, sin_rad, cos_rad)

      if help_exists:
        add_texts_upper_left(
          rotated,
          [
            "[rotate]",
            "select degree",
            "now: {0:.2f}s".format(frame_now / fps),
            "deg: {0}".format(degree_total),
          ],
        )
        add_texts_lower_right(
          rotated, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(movie, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree_total))
        cv2.destroyAllWindows()
        return float(degree_total)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


class RotatingPicture:
  """class to rotate picture"""

  def __init__(
    self,
    *,
    picture_list: Optional[List[str]] = None,
    is_colored: bool = False,
    degree: Optional[float] = None,
  ):
    """constructor

    Args:
        picture_list (Optional[List[str]], optional): list of pictures, directories where pictures are stored. Defaults to None.
        is_colored (bool, optional): [description]. flag to output in color to False.
        degree (Optional[float], optional): degree of rotation. Defaults to None.
    """
    self.__picture_list = picture_list
    self.__is_colored = is_colored
    self.__degree = degree

  def get_picture_list(self) -> Optional[List[str]]:
    return self.__picture_list

  def is_colored(self) -> bool:
    return self.__is_colored

  def get_degree(self) -> Optional[float]:
    return self.__degree

  def execute(self):
    """rotating process

    Returns:
        Tuple[str, List[str]]: type of output, list of output (picture or directory) path names
    """
    if self.get_degree() is not None:
      degree = self.get_degree()
      rad = degree / 180.0 * numpy.pi
      sin_rad = numpy.absolute(numpy.sin(rad))
      cos_rad = numpy.absolute(numpy.cos(rad))

    output_path_list = get_output_path(self.get_picture_list(), "rotated")
    clean_output_directory(output_path_list)
    return_list: List[str] = []

    for picture, output_path in zip(self.get_picture_list(), output_path_list):

      picture_path = pathlib.Path(picture)

      if picture_path.is_file():

        img = cv2.imread(picture)
        if self.get_degree() is None:
          degree = self.select_degree_picture(picture, img)
          if degree is None:
            continue
          rad = degree / 180.0 * numpy.pi
          sin_rad = numpy.absolute(numpy.sin(rad))
          cos_rad = numpy.absolute(numpy.cos(rad))

        print("rotating picture '{0}'...".format(picture))
        rotated = get_rotated_image(img, degree, sin_rad, cos_rad)
        name = str(pathlib.Path(output_path / picture_path.name))
        return_list.append(name)
        cv2.imwrite(
          name,
          rotated if self.is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY),
        )

      elif picture_path.is_dir():

        print("rotating picture in '{0}'...".format(picture))
        p_list = list(picture_path.iterdir())

        if self.get_degree() is None:
          degree = self.select_degree_directory(picture_path.name, p_list)
          if degree is None:
            continue
          rad = degree / 180.0 * numpy.pi
          sin_rad = numpy.absolute(numpy.sin(rad))
          cos_rad = numpy.absolute(numpy.cos(rad))

        return_list.append(str(output_path))

        for p in list(picture_path.iterdir()):
          img = cv2.imread(str(p))
          rotated = get_rotated_image(img, degree, sin_rad, cos_rad)
          cv2.imwrite(
            str(pathlib.Path(output_path / p.name)),
            rotated if self.is_colored() else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY),
          )

    return ("picture", return_list) if return_list else (None, [])

  def select_degree_picture(self, picture: str, img: numpy.array) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        picture (str): name of image
        img (numpy.array): cv2 image object

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(picture, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(picture, 500, 700)
    help_exists = True

    rotation_tick = 4
    rotation_tick_s = 90
    cv2.createTrackbar("degree\n", picture, 0, rotation_tick, no)
    cv2.createTrackbar("degree s\n", picture, 0, rotation_tick_s, no)

    print("--- rotate ---")
    print("select rotation degree in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      degree = cv2.getTrackbarPos("degree\n", picture) * rotation_tick_s
      degree_s = cv2.getTrackbarPos("degree s\n", picture)
      degree_total = degree + degree_s
      rad = degree_total / 180.0 * numpy.pi
      sin_rad = numpy.absolute(numpy.sin(rad))
      cos_rad = numpy.absolute(numpy.cos(rad))
      rotated = get_rotated_image(img, degree_total, sin_rad, cos_rad)

      if help_exists:
        add_texts_upper_left(
          rotated, ["[rotate]", "select degree", "deg: {0}".format(degree_total)],
        )
        add_texts_lower_right(
          rotated, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(picture, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree_total))
        cv2.destroyAllWindows()
        return float(degree_total)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None

  def select_degree_directory(
    self, directory: str, picture_path_list: List[pathlib.Path]
  ) -> Optional[float]:
    """select(get) rotation degree using GUI window

    Args:
        directory (str): directory name
        picture_path_list (List[pathlib.Path]): picture path list

    Returns:
        Optional[float]: rotation degree
    """
    cv2.namedWindow(directory, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(directory, 500, 700)
    help_exists = True

    frames = len(picture_path_list) - 1
    division = 50
    tick = division if division < frames else frames
    tick_s = (int)(frames / division) + 1
    cv2.createTrackbar("frame\n", directory, 0, tick - 1, no)
    cv2.createTrackbar("frame s\n", directory, 0, tick_s, no)

    rotation_tick = 4
    rotation_tick_s = 90
    cv2.createTrackbar("degree\n", directory, 0, rotation_tick, no)
    cv2.createTrackbar("degree s\n", directory, 0, rotation_tick_s, no)

    print("--- rotate ---")
    print("select rotation degree in GUI window!")
    print("(s: save if selected, h:on/off help, q/esc: abort)")

    while True:

      frame = cv2.getTrackbarPos("frame\n", directory) * tick_s
      frame_s = cv2.getTrackbarPos("frame s\n", directory)
      frame_now = frame + frame_s if frame + frame_s < frames else frames
      img = cv2.imread(str(picture_path_list[frame_now]))

      degree = cv2.getTrackbarPos("degree\n", directory) * rotation_tick_s
      degree_s = cv2.getTrackbarPos("degree s\n", directory)
      degree_total = degree + degree_s
      rad = degree_total / 180.0 * numpy.pi
      sin_rad = numpy.absolute(numpy.sin(rad))
      cos_rad = numpy.absolute(numpy.cos(rad))
      rotated = get_rotated_image(img, degree_total, sin_rad, cos_rad)

      if help_exists:
        add_texts_upper_left(
          rotated,
          [
            "[rotate]",
            "select degree",
            "frame: {0}".format(frame_now),
            "deg: {0}".format(degree_total),
          ],
        )
        add_texts_lower_right(
          rotated, ["s:save if selected", "h:on/off help", "q/esc:abort"]
        )

      cv2.imshow(directory, rotated)
      k = cv2.waitKey(1) & 0xFF

      if k == ord("s"):
        print("'s' is pressed. degree is saved ({0})".format(degree_total))
        cv2.destroyAllWindows()
        return float(degree_total)

      elif k == ord("h"):
        if help_exists:
          help_exists = False
        else:
          help_exists = True
        continue

      elif k == ord("q"):
        cv2.destroyAllWindows()
        print("'q' is pressed. abort")
        return None

      elif k == 27:
        cv2.destroyAllWindows()
        print("'Esq' is pressed. abort")
        return None


def get_rotated_image(
  img: numpy.array, degree: float, sin_rad: float, cos_rad: float
) -> numpy.array:
  """get rotated cv2 object

  Args:
      img (numpy.array): cv2 image object before rotation
      degree (float): degree of rotation
      sin_rad (float): sin of radian converted from degree
      cos_rad (float): sin of radian converted from degree

  Returns:
      numpy.array: cv2 image object after rotation
  """
  W, H = img.shape[1], img.shape[0]
  center = (W / 2, H / 2)
  W_rot = int(numpy.round(H * sin_rad + W * cos_rad))
  H_rot = int(numpy.round(H * cos_rad + W * sin_rad))
  size_rot = (W_rot, H_rot)
  rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
  affine_matrix = rotation_matrix.copy()
  affine_matrix[0][2] = affine_matrix[0][2] - W / 2 + W_rot / 2
  affine_matrix[1][2] = affine_matrix[1][2] - H / 2 + H_rot / 2
  return cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
