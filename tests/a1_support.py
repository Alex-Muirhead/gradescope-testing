from pathlib import Path


HELP = """
    The available commands are:

    'h' - provide help message
    'r' - read input paramaters for various scenarios from a file
    'p no_cols sep' - print a table with the number of columns specified in
                      no_cols and using "sep" as the column border
    'q' - quit
"""

MAIN_PROMPT = "Please enter a command: "
INVALID = "Invalid command!"


def load_data(directory: str, file_name: str) -> tuple[tuple[float | int, ...], ...]:
    """Load the parameter data from a specified file.

    Parameters
    ----------
    directory: str
        The name of the directory in which the file is located
    file_name: str
        The name of the file

    Returns
    -------
    tuple[tuple[float|int, ...], ...]
        A nested tuple containing the numbers for parameters
    """
    rel_path = Path(directory) / file_name

    if not rel_path.exists() or rel_path.is_dir():
        print("Must point to a valid parameter file!")
        return tuple()

    data = []
    with rel_path.open() as file:
        for raw_line in file.readlines():
            # Skip blank lines or comments
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith('#'):
                continue
            chunk = tuple(map(float, map(str.strip, raw_line.split(','))))
            data.append(chunk)

    return tuple(data)
