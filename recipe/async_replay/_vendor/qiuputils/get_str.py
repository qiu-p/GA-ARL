def dict2str(dictionary, indent_level=1):
    """dict to string for printing options.

    Args:
        dictionary (dict): dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): dictionary string for printing.
    """
    str_get = ''
    for k, v in dictionary.items():
        if isinstance(v, dict):
            str_get += ' ' * (indent_level * 2) + str(k) + ':[\n'
            str_get += dict2str(v, indent_level + 1)
            str_get += ' ' * (indent_level * 2) + ']\n'
        else:
            str_get += ' ' * (indent_level * 2) + str(k) + ': ' + str(v) + '\n'
    return str_get

def list2str(list: list) -> str:
    '''
    the values of list are supposed to be primitive data types
    '''
    if len(list) == 0:
        return '[]'
    else:
        str_get = '['
        for item in list[:-1]:
            str_get += str(item)
            str_get += ', '
        str_get += str(item)
        str_get += ']'
        return str_get

