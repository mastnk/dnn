import glob, os
exec( "".join(['from .{} import *\n'.format(os.path.splitext(os.path.basename(file))[0]) for file in glob.glob(os.path.join(os.path.dirname(__file__), '[a-zA-Z0-9]*.py'))]) )


'''
import glob
import os

# https://qiita.com/suzuki-kei/items/8fea67655abf216a5013
files = glob.glob(os.path.join(os.path.dirname(__file__), '[a-zA-Z0-9]*.py'))

import_cmd = ''
for file in files:
    #https://note.nkmk.me/python-os-basename-dirname-split-splitext/
    name = os.path.splitext(os.path.basename(file))[0]
    import_cmd += 'from .{} import *\n'.format(name)
exec( import_cmd )
'''
