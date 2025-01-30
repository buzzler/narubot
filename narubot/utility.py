def get_current_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

get_current_time_tool = {
    'type': 'function',
    'function': {
        'name': 'get_current_time',
        'description': 'Get the current time',
        'parameters': {
            'type': 'object',
            'required': [],
            'properties': {},
        },
    },
}

def get_tools():
    return [get_current_time_tool]

available_functions = {
    "get_current_time": get_current_time,
}
