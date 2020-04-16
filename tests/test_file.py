import pathlib
import pytest
from ma_file_handling import file


@pytest.fixture
def tmp_file_1():
  p = pathlib.Path("tmp_1.txt")
  p.touch()
  yield str(p)
  if p.is_file():
    p.unlink()


@pytest.mark.parametrize("arg", [1, None, ""])
def test_File_init_error(arg):
  with pytest.raises(TypeError):
    file.File(arg)


@pytest.fixture(params=["1", "2"])
def arg_remove(request):
  if request.param == "1":
    return (request.getfixturevalue("tmp_file_1"), True)
  elif request.param == "2":
    return ("tmp", False)


def test_File_remove(arg_remove):
  arg, expected = arg_remove
  f = file.File(arg)
  assert f.remove() == expected
