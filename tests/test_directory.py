import pathlib
import shutil
import pytest
from ma_file_handling import directory as dir


@pytest.fixture
def tmp_dir_1():
  p = pathlib.Path("tmp_1")
  p.mkdir(exist_ok=True)
  yield str(p)
  if p.is_dir():
    shutil.rmtree(p)


@pytest.fixture
def tmp_dir_2():
  p = pathlib.Path("tmp_2")
  p.mkdir(exist_ok=True)
  f = pathlib.Path(p / "tmp.txt")
  f.touch()
  yield str(p)
  if p.is_dir():
    shutil.rmtree(p)


@pytest.mark.parametrize("arg", [1, None, ""])
def test_File_init_error(arg):
  with pytest.raises(TypeError):
    dir.Directory(arg)


@pytest.fixture(params=["1", "2", "3"])
def arg_clean(request):
  if request.param == "1":
    return (request.getfixturevalue("tmp_dir_1"), True)
  elif request.param == "2":
    return (request.getfixturevalue("tmp_dir_2"), True)
  elif request.param == "3":
    return ("tmp", False)


def test_Directory_clean(arg_clean):
  arg, expected = arg_clean
  d = dir.Directory(arg)
  assert d.clean() == expected


@pytest.fixture(params=["1", "2"])
def arg_remove(request):
  if request.param == "1":
    return (request.getfixturevalue("tmp_dir_1"), True)
  elif request.param == "2":
    return ("tmp", False)


def test_Directory_remove(arg_remove):
  arg, expected = arg_remove
  d = dir.Directory(arg)
  assert d.remove() == expected
