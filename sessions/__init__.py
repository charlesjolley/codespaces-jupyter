"""Load demo sessions"""

import yaml
from os import path


def _load_yaml(filename):
    filepath = path.join(path.dirname(__file__), filename)
    fd = open(filepath, mode="r", encoding="utf8")
    return yaml.load(fd, yaml.Loader)


plan_date = _load_yaml('plan_date.yaml')
meeting_counseling = _load_yaml('meeting_counseling.yaml')
travel = _load_yaml('travel.yaml')
