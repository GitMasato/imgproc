"""cli command for image process functions.

Basic required arguments are list of movies or pictures. if no movie or picture is
given, this program will be terminated

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

see usage '-h option'

"""
import argparse
import sys
from typing import List
from image_process import process


def call_check_arg(
  arg_name: str, args: argparse.Namespace, parser: argparse.ArgumentParser
):
  """check if every cli-arguments is correct

  Args:
      arg_name (str): subcommand name
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Raises:
      exception.ArgMissingError: when given cli-argument is not enough
  """
  items = [
    value for key, value in args.__dict__.items() if key != "call" and key != "color"
  ]
  if not [item for item in items if item is not None]:
    sys.exit(parser.parse_args([arg_name, "--help"]))

  if hasattr(args, "movie") and hasattr(args, "picture"):
    if not args.movie and not args.picture:
      sys.exit("no movie and picture is given!")
  elif hasattr(args, "movie"):
    if not args.movie:
      sys.exit("no movie is given!")
  elif hasattr(args, "picture"):
    if not args.picture:
      sys.exit("no picture is given!")


def call_animate(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when animate command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("animate", args, parser)
  parameter = process.AnimatingParameter(args.picture, args.color, args.fps)
  return process.ProcessExecution(process.AnimatingPicture(parameter))


def call_binarize(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when binarize command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("binarize", args, parser)
  p: List[process.ABCProcess] = []
  if args.picture:
    parameter = process.BinarizingParameter(args.picture, args.threshold)
    p.append(process.BinarizingPicture(parameter))
  if args.movie:
    parameter = process.BinarizingParameter(args.movie, args.threshold)
    p.append(process.BinarizingMovie(parameter))
  return process.ProcessesExecution(p)


def call_capture(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when capture command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("capture", args, parser)
  parameter = process.CapturingParameter(args.movie, args.color, args.time)
  return process.ProcessExecution(process.CapturingMovie(parameter))


def call_crop(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when crop command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("crop", args, parser)
  p: List[process.ABCProcess] = []
  if args.picture:
    parameter = process.CroppingParameter(args.picture, args.color, args.position)
    p.append(process.CroppingPicture(parameter))
  if args.movie:
    parameter = process.CroppingParameter(args.movie, args.color, args.position)
    p.append(process.CroppingMovie(parameter))
  return process.ProcessesExecution(p)


def call_hist_luminance(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when hist_luminance command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("hist-luminance", args, parser)
  parameter = process.CreatingLuminanceHistgramParameter(args.picture, args.color)
  return process.ProcessExecution(process.CreatingLuminanceHistgramPicture(parameter))


def call_resize(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when resize command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("resize", args, parser)
  p: List[process.ABCProcess] = []
  if args.picture:
    parameter = process.ResizingParameter(args.picture, args.color, args.scale)
    p.append(process.ResizingPicture(parameter))
  if args.movie:
    parameter = process.ResizingParameter(args.movie, args.color, args.scale)
    p.append(process.ResizingMovie(parameter))
  return process.ProcessesExecution(p)


def call_rotate(
  args: argparse.Namespace, parser: argparse.ArgumentParser
) -> process.ABCProcessExecution:
  """call function when rotate command is given

  Args:
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  call_check_arg("rotate", args, parser)
  p: List[process.ABCProcess] = []
  if args.picture:
    parameter = process.RotatingParameter(args.picture, args.color, args.degree)
    p.append(process.RotatingPicture(parameter))
  if args.movie:
    parameter = process.RotatingParameter(args.movie, args.color, args.degree)
    p.append(process.RotatingMovie(parameter))
  return process.ProcessesExecution(p)


def read_cli_argument() -> process.ABCProcessExecution:
  """read and parse cli arguments

  Returns:
      process.ABCProcessExecution: process.ABCProcessExecution object
  """
  parser = argparse.ArgumentParser(
    prog="image_process.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description="python package providing functions of image process."
    + "\noutput data is generated in 'cv2' directory under current location.",
  )
  subparsers = parser.add_subparsers()

  def add_argument_movie(parser: argparse.ArgumentParser):
    parser.add_argument(
      "--movie",
      nargs="*",
      type=str,
      default=None,
      metavar="path",
      help="movie path" + "\n ",
    )

  def add_argument_picture(parser: argparse.ArgumentParser):
    parser.add_argument(
      "--picture",
      nargs="*",
      type=str,
      default=None,
      metavar="path",
      help="path of picture or directory where pictures are stored"
      + "\nIf directory name is given,"
      + "same process will be applied to all pictures in the directory"
      + "\n ",
    )

  def add_argument_color(parser: argparse.ArgumentParser):
    parser.add_argument(
      "--color",
      action="store_true",
      help="to output in color (default=false (gray))" + "\n ",
    )

  parser_animate = subparsers.add_parser(
    "animate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to crate movie (mp4) from pictures"
    + "\n(see sub-option 'animate -h')"
    + "\n ",
    description="sub-command 'animate': to crate movie (mp4) from pictures",
  )
  add_argument_picture(parser_animate)
  add_argument_color(parser_animate)
  parser_animate.set_defaults(call=call_animate)
  parser_animate.add_argument(
    "--fps",
    type=float,
    default=None,
    metavar="fps",
    help="fps of created movie (float) default=20.0" + "\n ",
  )

  parser_binarize = subparsers.add_parser(
    "binarize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to binarize picture/movie" + "\n(see sub-option 'binarize -h')" + "\n ",
    description="sub-command 'binarize': to binarize picture/movie",
  )
  add_argument_movie(parser_binarize)
  add_argument_picture(parser_binarize)
  parser_binarize.set_defaults(call=call_binarize)
  parser_binarize.add_argument(
    "--threshold",
    nargs=2,
    type=int,
    default=None,
    metavar=("low", "high"),
    help="thresholds of gray-scale luminance (int) [0-255]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_capture = subparsers.add_parser(
    "capture",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to capture movie" + "\n(see sub-option 'capture -h')" + "\n ",
    description="sub-command 'capture': to capture movie",
  )
  add_argument_movie(parser_capture)
  add_argument_color(parser_capture)
  parser_capture.set_defaults(call=call_capture)
  parser_capture.add_argument(
    "--time",
    nargs=3,
    type=float,
    default=None,
    metavar=("start", "stop", "step"),
    help="time at the beginning and end of capture"
    + "\nand the time step (float) [s]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_crop = subparsers.add_parser(
    "crop",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to crop movie/picture" + "\n(see sub-option 'crop -h')" + "\n ",
    description="sub-command 'crop': to crop movie/picture",
  )
  add_argument_movie(parser_crop)
  add_argument_picture(parser_crop)
  add_argument_color(parser_crop)
  parser_crop.set_defaults(call=call_crop)
  parser_crop.add_argument(
    "--position",
    nargs=4,
    type=int,
    default=None,
    metavar=("x_1", "y_1", "x_2", "y_2"),
    help="position of cropping (int) [pixel, 1 < 2]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_hist_luminance = subparsers.add_parser(
    "hist-luminance",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to create luminance histgram of picture"
    + "\n(see sub-option 'hist-luminance -h')"
    + "\n ",
    description="sub-command 'hist_luminance': to create luminance histgram of picture",
  )
  add_argument_picture(parser_hist_luminance)
  add_argument_color(parser_hist_luminance)
  parser_hist_luminance.set_defaults(call=call_hist_luminance)

  parser_resize = subparsers.add_parser(
    "resize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to resize movie/picture" + "\n(see sub-option 'resize -h')" + "\n ",
    description="sub-command 'resize': to resize movie/picture",
  )
  add_argument_movie(parser_resize)
  add_argument_picture(parser_resize)
  add_argument_color(parser_resize)
  parser_resize.set_defaults(call=call_resize)
  parser_resize.add_argument(
    "--scale",
    nargs=2,
    type=float,
    default=None,
    metavar=("x", "y"),
    help="scaling ratio (float) default=(1.0,1.0)" + "\n ",
  )

  parser_rotate = subparsers.add_parser(
    "rotate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to rotate movie/picture" + "\n(see sub-option 'rotate -h')",
    description="sub-command 'rotate': to rotate movie/picture",
  )
  add_argument_movie(parser_rotate)
  add_argument_picture(parser_rotate)
  add_argument_color(parser_rotate)
  parser_rotate.set_defaults(call=call_rotate)
  parser_rotate.add_argument(
    "--degree",
    type=float,
    default=None,
    metavar=("degree"),
    help="degree of rotation (float) [degree] default=0.0" + "\n ",
  )

  if len(sys.argv) <= 1:
    sys.exit(parser.format_help())

  args = parser.parse_args()
  return args.call(args, parser)


def main() -> None:
  """cli command main function
  """
  image_process = read_cli_argument()
  image_process.execute()


if __name__ == "__main__":
  main()
