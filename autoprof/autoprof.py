

def GetOptions(c):
    """
    Extract all of the AutoProf user optionional parameters form the config file.
    User options are identified as any python object that starts with "ap\_" in the
    variable name.
    """
    newoptions = {}
    for var in dir(c):
        if var.startswith("ap_"):
            val = getattr(c, var)
            if not val is None:
                newoptions[var] = val

    return newoptions
