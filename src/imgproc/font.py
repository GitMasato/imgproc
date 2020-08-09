"""font module to manage available fonts
"""
import pprint
from matplotlib import font_manager
from typing import Dict, List, Optional


class AvailableFontManager:
  """class of managing available font information"""

  @staticmethod
  def display_font_list():
    f = sorted(list(set([f.name for f in font_manager.fontManager.ttflist])))
    pprint.pprint(f, width=80, compact=True)

  def __init__(self):
    """constructor"""
    self.__font_dict = {f.name: f.fname for f in font_manager.fontManager.ttflist}

  def get_font_dict(self) -> Dict:
    return self.__font_dict

  def get_font_list(self) -> List[str]:
    return self.__font_dict.keys()

  def get_font_list_sorted(self) -> List[str]:
    return sorted(self.__font_dict.keys())

  def get_font_file_list(self) -> List[str]:
    return self.__font_dict.values()

  def get_font_file_list_sorted(self) -> List[str]:
    return sorted(self.__font_dict.values())

  def get_font_file(self, name: str) -> Optional[str]:
    return self.__font_dict[name] if name in self.__font_dict else None
