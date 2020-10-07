"""cli command for image process functions.

Required arguments are list of movies or pictures. if no movie or picture is given, this program will be terminated.

Output data (after image process) will be generated in 'cv2/target-noExtension/process-name/target' directory under current location (e.g. ./cv2/test/binarized/test.png). if movie/picture in 'cv2' directory is given as input (e.g. ./cv2/test/rotated/test.png), the output will be generated in same cv2 but 'selected process' directory (e.g. ./cv2/test/binarized/test.png)

Examples:
  imgproc animate --picture tmp_dir --fps 20.0 --color

see usage '-h option'

"""
import argparse
import sys
from imgproc import process, font


def call_base(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when no sub-command is given
  """
  if args.available_font:
    font.AvailableFontManager.display_font_list()
    sys.exit()

  if not check_arg(args):
    sys.exit(parser.parse_args(["--help"]))


def call_animate(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when animate sub-command is given
  """
  check_arg_subcommand("animate", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.AnimatingMovie(
      target_list=m_list, is_colored=args.color, fps=args.fps
    ).execute()

  if p_list:
    process.AnimatingPicture(
      target_list=p_list, is_colored=args.color, fps=args.fps
    ).execute()

  if d_list:
    process.AnimatingPictureDirectory(
      target_list=d_list, is_colored=args.color, fps=args.fps
    ).execute()


def call_binarize(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when binarize sub-command is given
  """
  check_arg_subcommand("binarize", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.BinarizingMovie(target_list=m_list, thresholds=args.threshold).execute()

  if p_list:
    process.BinarizingPicture(target_list=p_list, thresholds=args.threshold).execute()

  if d_list:
    process.BinarizingPictureDirectory(
      target_list=d_list, thresholds=args.threshold
    ).execute()


def call_capture(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when capture sub-command is given
  """
  check_arg_subcommand("capture", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list:
    sys.exit("no movie is given!")

  if m_list:
    process.CapturingMovie(
      target_list=m_list, is_colored=args.color, times=args.time
    ).execute()


def call_concatenate(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when concatenate sub-command is given
  """
  check_arg_subcommand("concatenate", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.ConcatenatingMovie(
      target_list=m_list, is_colored=args.color, number_x=args.number_x
    ).execute()

  if p_list:
    process.ConcatenatingPicture(
      target_list=p_list, is_colored=args.color, number_x=args.number_x
    ).execute()

  if d_list:
    process.ConcatenatingPictureDirectory(
      target_list=d_list, is_colored=args.color, number_x=args.number_x
    ).execute()


def call_crop(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when crop sub-command is given
  """
  check_arg_subcommand("crop", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.CroppingMovie(
      target_list=m_list, is_colored=args.color, positions=args.position
    ).execute()

  if p_list:
    process.CroppingPicture(
      target_list=p_list, is_colored=args.color, positions=args.position
    ).execute()

  if d_list:
    process.CroppingPictureDirectory(
      target_list=d_list, is_colored=args.color, positions=args.position
    ).execute()


def call_hist_luminance(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when hist_luminance sub-command is given
  """
  check_arg_subcommand("hist-luminance", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not p_list and not d_list:
    sys.exit("no picture, directory is given!")

  if p_list:
    process.CreatingLuminanceHistgramPicture(
      target_list=p_list, is_colored=args.color
    ).execute()

  if d_list:
    process.CreatingLuminanceHistgramPictureDirectory(
      target_list=d_list, is_colored=args.color
    ).execute()


def call_resize(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when resize sub-command is given
  """
  check_arg_subcommand("resize", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.ResizingMovie(
      target_list=m_list, is_colored=args.color, scales=args.scale
    ).execute()

  if p_list:
    process.ResizingPicture(
      target_list=p_list, is_colored=args.color, scales=args.scale
    ).execute()

  if d_list:
    process.ResizingPictureDirectory(
      target_list=d_list, is_colored=args.color, scales=args.scale
    ).execute()


def call_rotate(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when rotate sub-command is given
  """
  check_arg_subcommand("rotate", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.RotatingMovie(
      target_list=m_list, is_colored=args.color, degree=args.degree
    ).execute()

  if p_list:
    process.RotatingPicture(
      target_list=p_list, is_colored=args.color, degree=args.degree
    ).execute()

  if d_list:
    process.RotatingPictureDirectory(
      target_list=d_list, is_colored=args.color, degree=args.degree
    ).execute()


def call_subtitle(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when subtitle sub-command is given
  """
  check_arg_subcommand("subtitle", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list and not p_list and not d_list:
    sys.exit("no movie, picture, directory is given!")

  if m_list:
    process.SubtitlingMovie(
      target_list=m_list,
      is_colored=args.color,
      text=args.text,
      position=args.position,
      size=args.size,
      time=args.time,
    ).execute()

  if p_list:
    process.SubtitlingPicture(
      target_list=p_list,
      is_colored=args.color,
      text=args.text,
      position=args.position,
      size=args.size,
    ).execute()

  if d_list:
    process.SubtitlingPictureDirectory(
      target_list=d_list,
      is_colored=args.color,
      text=args.text,
      position=args.position,
      size=args.size,
    ).execute()


def call_trim(args: argparse.Namespace, parser: argparse.ArgumentParser):
  """call function when trim sub-command is given
  """
  check_arg_subcommand("trim", args, parser)
  m_list, p_list, d_list = process.sort_target_type(args.target)

  if not m_list:
    sys.exit("no movie is given!")

  if m_list:
    process.TrimmingMovie(
      target_list=m_list, is_colored=args.color, times=args.time
    ).execute()


def check_arg(args: argparse.Namespace):
  """check if required cli-arguments are given

  Args:
      args (argparse.Namespace): argparse.Namespace object

  Returns:
      bool: False if arguments are not enough
  """
  arg_list = [value for key, value in args.__dict__.items() if (key != "call")]

  if not [arg for arg in arg_list if (arg is not None) and (arg is not False)]:
    return False
  else:
    return True


def check_arg_subcommand(
  arg_name: str, args: argparse.Namespace, parser: argparse.ArgumentParser
):
  """check if required cli-arguments are given

  Args:
      arg_name (str): subcommand name
      args (argparse.Namespace): argparse.Namespace object
      parser (argparse.ArgumentParser): argparse.ArgumentParser object
  """
  if not check_arg(args):
    sys.exit(parser.parse_args([arg_name, "--help"]))

  if hasattr(args, "target"):
    if not args.target:
      sys.exit("no target is given!")


def add_argument_target(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--target",
    nargs="*",
    type=str,
    default=None,
    metavar="path",
    help="path of movie, picture, or directory where pictures are stored\n"
    + "If directory name is given,\n"
    + "same process will be applied to all pictures in the directory\n ",
  )


def add_argument_target_only_movie(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--target",
    nargs="*",
    type=str,
    default=None,
    metavar="path",
    help="path of movie\n ",
  )


def add_argument_available_font(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--available-font",
    action="store_true",
    help="to see available fonts in this program\n ",
  )


def add_argument_font_type(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--font-type",
    type=str,
    default=None,
    metavar="font",
    help="font type to be used. see available fonts '--available-font'\n",
  )


def add_argument_color(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--color", action="store_true", help="to output in color (default=false (gray))\n ",
  )


def cli_execution():
  """read, parse, and execute cli arguments
  """
  parser = argparse.ArgumentParser(
    prog="imgproc.py",
    formatter_class=argparse.RawTextHelpFormatter,
    description="python package providing functions of image process.\n"
    + "Output data is generated in 'cv2' directory under current location.\n\n"
    + "Examples:\n"
    + "  imgproc animate --target tmp_dir --fps 20.0 --color\n",
  )
  add_argument_available_font(parser)
  parser.set_defaults(call=call_base)
  subparsers = parser.add_subparsers()

  parser_animate = subparsers.add_parser(
    "animate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to create movie (mp4) from input file\n(see sub-option 'animate -h')\n ",
    description="sub-command 'animate': to create movie (mp4) from input file",
  )
  add_argument_target(parser_animate)
  add_argument_color(parser_animate)
  parser_animate.set_defaults(call=call_animate)
  parser_animate.add_argument(
    "--fps",
    type=float,
    default=None,
    metavar="fps",
    help="fps of created movie (float)\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_binarize = subparsers.add_parser(
    "binarize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to binarize picture/movie\n(see sub-option 'binarize -h')\n ",
    description="sub-command 'binarize': to binarize picture/movie",
  )
  add_argument_target(parser_binarize)
  parser_binarize.set_defaults(call=call_binarize)
  parser_binarize.add_argument(
    "--threshold",
    nargs=2,
    type=int,
    default=None,
    metavar=("low", "high"),
    help="thresholds of gray-scale luminance (int) [0-255]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_capture = subparsers.add_parser(
    "capture",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to capture movie, creating pictures\n(see sub-option 'capture -h')\n ",
    description="sub-command 'capture': to capture movie, creating pictures",
  )
  add_argument_target_only_movie(parser_capture)
  add_argument_color(parser_capture)
  parser_capture.set_defaults(call=call_capture)
  parser_capture.add_argument(
    "--time",
    nargs=3,
    type=float,
    default=None,
    metavar=("start", "stop", "step"),
    help="time at beginning and end of capture and time step (float) [s]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_concatenate = subparsers.add_parser(
    "concatenate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to concatenate movie/picture\n(see sub-option 'crop -h')\n ",
    description="sub-command 'concatenate': to concatenate movie/picture\n\n"
    + "max number of targets is 25.\n"
    + "sizes of pictures are adjusted based on first target size.\n",
  )
  parser_concatenate.add_argument(
    "--target",
    nargs="*",
    type=str,
    default=None,
    metavar="path",
    help="path of movie, picture, or directory where pictures are stored\n"
    + "If directory name is given,\n"
    + "pictures in the directory will be concatenated\n ",
  )
  add_argument_color(parser_concatenate)
  parser_concatenate.set_defaults(call=call_concatenate)
  parser_concatenate.add_argument(
    "--number-x",
    type=int,
    default=None,
    metavar=("x"),
    help="number of targets concatenated in x direction. max is 5.\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_crop = subparsers.add_parser(
    "crop",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to crop movie/picture\n(see sub-option 'crop -h')\n ",
    description="sub-command 'crop': to crop movie/picture",
  )
  add_argument_target(parser_crop)
  add_argument_color(parser_crop)
  parser_crop.set_defaults(call=call_crop)
  parser_crop.add_argument(
    "--position",
    nargs=4,
    type=int,
    default=None,
    metavar=("x_1", "y_1", "x_2", "y_2"),
    help="position of cropping (int) [pixel, 1 < 2]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_hist_luminance = subparsers.add_parser(
    "hist-luminance",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to create luminance histgram of picture\n"
    + "(see sub-option 'hist-luminance -h')\n ",
    description="sub-command 'hist_luminance': to create luminance histgram of picture",
  )
  parser_hist_luminance.add_argument(
    "--target",
    nargs="*",
    type=str,
    default=None,
    metavar="path",
    help="path of picture or directory where pictures are stored\n"
    + "If directory name is given,\n"
    + "same process will be applied to all pictures in the directory\n ",
  )
  add_argument_color(parser_hist_luminance)
  parser_hist_luminance.set_defaults(call=call_hist_luminance)

  parser_resize = subparsers.add_parser(
    "resize",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to resize movie/picture\n(see sub-option 'resize -h')\n ",
    description="sub-command 'resize': to resize movie/picture",
  )
  add_argument_target(parser_resize)
  add_argument_color(parser_resize)
  parser_resize.set_defaults(call=call_resize)
  parser_resize.add_argument(
    "--scale",
    nargs=2,
    type=float,
    default=None,
    metavar=("x", "y"),
    help="scaling ratio (float) [x,y]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_rotate = subparsers.add_parser(
    "rotate",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to rotate movie/picture\n(see sub-option 'rotate -h')\n ",
    description="sub-command 'rotate': to rotate movie/picture",
  )
  add_argument_target(parser_rotate)
  add_argument_color(parser_rotate)
  parser_rotate.set_defaults(call=call_rotate)
  parser_rotate.add_argument(
    "--degree",
    type=float,
    default=None,
    metavar=("degree"),
    help="degree of rotation (float) [degree]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  parser_subtitle = subparsers.add_parser(
    "subtitle",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to subtitle movie/picture\n(see sub-option 'subtitle -h')\n ",
    description="sub-command 'subtitle': to subtitle movie/picture",
  )
  add_argument_target(parser_subtitle)
  add_argument_color(parser_subtitle)
  parser_subtitle.set_defaults(call=call_subtitle)
  parser_subtitle.add_argument(
    "--time",
    nargs=2,
    type=float,
    default=None,
    metavar=("start", "stop"),
    help="time at beginning and end of subtitling (float) [s]\n"
    + "this argument is neglected for picture or directory\n"
    + "if this is not given, you will select this in GUI window\n ",
  )
  parser_subtitle.add_argument(
    "--position",
    nargs=2,
    type=int,
    default=None,
    metavar=("x", "y"),
    help="position of left end of text (int) [pixel]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )
  parser_subtitle.add_argument(
    "--size",
    type=float,
    default=None,
    metavar=("size"),
    help="size of text\n"
    + "if this is not given, you will select this in GUI window\n ",
  )
  parser_subtitle.add_argument(
    "--text",
    type=str,
    default=None,
    metavar=("text"),
    help="text to be added into movie/picture\n ",
  )

  parser_trim = subparsers.add_parser(
    "trim",
    formatter_class=argparse.RawTextHelpFormatter,
    help="to trim movie (mp4)\n(see sub-option 'trim -h')\n ",
    description="sub-command 'trim': to trim movie (mp4)",
  )
  add_argument_target_only_movie(parser_trim)
  add_argument_color(parser_trim)
  parser_trim.set_defaults(call=call_trim)
  parser_trim.add_argument(
    "--time",
    nargs=2,
    type=float,
    default=None,
    metavar=("start", "stop"),
    help="time at beginning and end of trimming (float) [s]\n"
    + "if this is not given, you will select this in GUI window\n ",
  )

  args = parser.parse_args()
  args.call(args, parser)


def main() -> None:
  """cli command main function
  """
  cli_execution()


if __name__ == "__main__":
  main()
