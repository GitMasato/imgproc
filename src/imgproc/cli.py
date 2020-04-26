"""cli command for image process functions.

Required arguments are list of movies or pictures. if no movie or picture is given, this program will be terminated.

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

If directory path where pictures are stored is given as picture argument, same image-process will be applied to all pictures in the directory.

Examples:
  imgproc animate --picture tmp_dir --fps 20.0 --color

see usage '-h option'

"""
import argparse
import pathlib
import sys
from imgproc import process


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

  if hasattr(args, "target"):
    if not args.target:
      sys.exit("no target is given!")

  if hasattr(args, "type"):
    if args.type == "picture":
      for t in args.target:
        path = pathlib.Path(t)
        if path.is_dir():
          if not list(path.iterdir()):
            sys.exit("no picture exists in '{0}'!".format(str(path)))


def call_animate(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when animate command is given
  """
  call_check_arg("animate", args, parser)
  return process.AnimatingPicture(
    picture_list=args.target, is_colored=args.color, fps=args.fps
  )


def call_binarize(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when binarize command is given
  """
  call_check_arg("binarize", args, parser)
  if args.type == "picture":
    return process.BinarizingPicture(
      picture_list=args.target, thresholds=args.threshold
    )
  elif args.type == "movie":
    return process.BinarizingMovie(movie_list=args.target, thresholds=args.threshold)


def call_capture(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when capture command is given
  """
  call_check_arg("capture", args, parser)
  return process.CapturingMovie(
    movie_list=args.target, is_colored=args.color, times=args.time
  )


def call_crop(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when crop command is given
  """
  call_check_arg("crop", args, parser)
  if args.type == "picture":
    return process.CroppingPicture(
      picture_list=args.target, is_colored=args.color, positions=args.position
    )
  elif args.type == "movie":
    return process.CroppingMovie(
      movie_list=args.target, is_colored=args.color, positions=args.position
    )


def call_hist_luminance(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when hist_luminance command is given
  """
  call_check_arg("hist-luminance", args, parser)
  return process.CreatingLuminanceHistgramPicture(
    picture_list=args.target, is_colored=args.color
  )


def call_resize(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when resize command is given
  """
  call_check_arg("resize", args, parser)
  if args.type == "picture":
    return process.ResizingPicture(
      picture_list=args.target, is_colored=args.color, scales=args.scale
    )
  elif args.type == "movie":
    return process.ResizingMovie(
      movie_list=args.target, is_colored=args.color, scales=args.scale
    )


def call_rotate(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when rotate command is given
  """
  call_check_arg("rotate", args, parser)
  if args.type == "picture":
    return process.RotatingPicture(
      picture_list=args.target, is_colored=args.color, degree=args.degree
    )
  elif args.type == "movie":
    return process.RotatingMovie(
      movie_list=args.target, is_colored=args.color, degree=args.degree
    )


def read_cli_argument() -> process.ABCProcess:
  """read and parse cli arguments

  Returns:
      process.ABCProcess: process.ABCProcess object
  """
  parser = argparse.ArgumentParser(
    prog="imgproc.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description="python package providing functions of image process.\n"
    + "Output data is generated in 'cv2' directory under current location.\n\n"
    + "Examples:\n"
    + "  imgproc animate --picture tmp_dir --fps 20.0 --color\n",
  )
  subparsers = parser.add_subparsers()

  def add_argument_type(parser: argparse.ArgumentParser):
    parser.add_argument(
      "--type", required=True, choices=["picture", "movie"], help="target type" + "\n ",
    )

  def add_argument_target(parser: argparse.ArgumentParser):
    parser.add_argument(
      "--target",
      nargs="*",
      type=str,
      default=None,
      metavar="path",
      help="path of movie, picture, or directory where pictures are stored"
      + "\nIf directory name is given,"
      + "\nsame process will be applied to all pictures in the directory"
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
  add_argument_target(parser_animate)
  add_argument_color(parser_animate)
  parser_animate.set_defaults(call=call_animate)
  parser_animate.add_argument(
    "--fps",
    type=float,
    default=None,
    metavar="fps",
    help="fps of created movie (float)"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_binarize = subparsers.add_parser(
    "binarize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to binarize picture/movie" + "\n(see sub-option 'binarize -h')" + "\n ",
    description="sub-command 'binarize': to binarize picture/movie",
  )
  add_argument_type(parser_binarize)
  add_argument_target(parser_binarize)
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
  add_argument_target(parser_capture)
  add_argument_color(parser_capture)
  parser_capture.set_defaults(call=call_capture)
  parser_capture.add_argument(
    "--time",
    nargs=3,
    type=float,
    default=None,
    metavar=("start", "stop", "step"),
    help="time at beginning and end of capture and time step (float) [s]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_crop = subparsers.add_parser(
    "crop",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to crop movie/picture" + "\n(see sub-option 'crop -h')" + "\n ",
    description="sub-command 'crop': to crop movie/picture",
  )
  add_argument_type(parser_crop)
  add_argument_target(parser_crop)
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
  add_argument_target(parser_hist_luminance)
  add_argument_color(parser_hist_luminance)
  parser_hist_luminance.set_defaults(call=call_hist_luminance)

  parser_resize = subparsers.add_parser(
    "resize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to resize movie/picture" + "\n(see sub-option 'resize -h')" + "\n ",
    description="sub-command 'resize': to resize movie/picture",
  )
  add_argument_type(parser_resize)
  add_argument_target(parser_resize)
  add_argument_color(parser_resize)
  parser_resize.set_defaults(call=call_resize)
  parser_resize.add_argument(
    "--scale",
    nargs=2,
    type=float,
    default=None,
    metavar=("x", "y"),
    help="scaling ratio (float) [x,y]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
  )

  parser_rotate = subparsers.add_parser(
    "rotate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to rotate movie/picture" + "\n(see sub-option 'rotate -h')",
    description="sub-command 'rotate': to rotate movie/picture",
  )
  add_argument_type(parser_rotate)
  add_argument_target(parser_rotate)
  add_argument_color(parser_rotate)
  parser_rotate.set_defaults(call=call_rotate)
  parser_rotate.add_argument(
    "--degree",
    type=float,
    default=None,
    metavar=("degree"),
    help="degree of rotation (float) [degree]"
    + "\nif this is not given, you will select this in GUI window"
    + "\n ",
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
