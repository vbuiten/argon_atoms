'''Utility function for easily handling data transfer.'''

import os

def defaultPath():
    '''Load whichever "default" path is available.'''

    path = "/net/vdesk/data2/buiten/COP/"

    if not os.access(path, os.F_OK):
        path = "C:\\Users\\victo\\Documents\\Uni\\COP\\"

    return path


def folderPath(folder, defaultbase=True):
    '''Make the new folder if necessary and return the path of the folder.
    The returned path is absolute if defaultbase=True.'''

    if defaultbase:
        basepath = defaultPath()
        folderpath = basepath+folder

    else:
        folderpath = folder

    if not os.access(folderpath, os.F_OK):
        os.mkdir(folderpath)

    return folderpath