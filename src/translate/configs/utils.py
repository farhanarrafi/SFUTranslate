from pathlib import Path


def get_resources_dir():
    """
    :return: an object of `pathlib.PosixPath` which can be directly opened or traversed
    """
    for path in Path.cwd().parents:
        if str(path).endswith("src"):
            return Path(path.parent, "resources")
    else:
        cwd = Path.cwd()
        if str(cwd).endswith("src"):
            return Path(cwd.parent, "resources")
        else:
            raise ValueError("Unable to find /src/ directory address!")


def get_resource_file(resource_name):
    """
    :param resource_name: The name of the resource file placed in `SFUTranslate/resources` directory
    :return: an object of `pathlib.PosixPath` which can be directly opened or traversed
    """
    return Path(get_resources_dir(), resource_name)