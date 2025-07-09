def parse_dict(value: str):
    return dict(pair.split('=') for pair in value.split(','))
