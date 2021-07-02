from . import Lectures
from . import Curriculum

import inspect

all_lectures = [getattr(Lectures, cls_name) for cls_name in dir(Lectures)]
all_lectures = [lect_cls() for lect_cls in all_lectures if inspect.isclass(lect_cls)
                and issubclass(lect_cls, Curriculum.Lecture) and lect_cls is not Curriculum.Lecture]

