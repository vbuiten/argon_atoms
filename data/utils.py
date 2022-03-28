'''Utility function for easily handling data transfer.'''

import os

def defaultPath():
    '''
    Load whichever "default" path is available.

    :return path: str
            Available default path
    '''

    path = "/net/vdesk/data2/buiten/COP/"

    if not os.access(path, os.F_OK):
        path = "C:\\Users\\victo\\Documents\\Uni\\COP\\"

    return path


def folderPath(folder, defaultbase=True):
    '''
    Make the new folder if necessary and return the path of the folder.
    The returned path is absolute if defaultbase=True.

    :param folder: str
            Name of the folder to use or create
    :param defaultbase: bool
            If True, uses the "default" base path (see defaultPath()). Default is True.
    :return folderpath: str
            Path of the folder, either absolute or relative to the current working directory
    '''

    if defaultbase:
        basepath = defaultPath()
        folderpath = basepath+folder

    else:
        folderpath = folder

    if not os.access(folderpath, os.F_OK):
        os.mkdir(folderpath)

    return folderpath