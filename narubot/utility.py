import os
import platform


def get_current_time():
    """Get the current time."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_system_info():
    """Get basic system information."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd(),
    }


get_current_time_tool = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}

get_system_info_tool = {
    "type": "function",
    "function": {
        "name": "get_system_info",
        "description": "Get basic system information like OS, hostname, Python version, and working directory",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


def get_tools():
    return [
        get_current_time_tool,
        get_system_info_tool,
    ]


available_functions = {
    "get_current_time": get_current_time,
    "get_system_info": get_system_info,
}
