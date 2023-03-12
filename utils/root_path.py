from os import path


def get_root_path():
    current_file_path = path.abspath(__file__)
    current_dir_path = path.dirname(current_file_path)
    project_root_path = path.dirname(current_dir_path)

    return project_root_path
