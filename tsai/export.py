# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/007_export.ipynb (unless otherwise specified).

__all__ = ['get_nb_name', 'get_colab_nb_name', 'get_nb_path', 'get_script_path', 'FILE_ERROR', 'CONN_ERROR', 'is_lab',
           'is_colab', 'to_local_time', 'maybe_mount_drive', 'nb2py']

# Cell
"""
Code copied from the great nbdev library: https://github.com/fastai/nbdev/blob/master/nbdev/export.py
"""

def _mk_flag_re(body, n_params, comment):
    "Compiles a regex for finding nbdev flags"
    import re
    assert body!=True, 'magics no longer supported'
    prefix = r"\s*\#\s*"
    param_group = ""
    if n_params == -1: param_group = r"[ \t]+(.+)"
    if n_params == 1: param_group = r"[ \t]+(\S+)"
    if n_params == (0,1): param_group = r"(?:[ \t]+(\S+))?"
    return re.compile(rf"""
# {comment}:
^            # beginning of line (since re.MULTILINE is passed)
{prefix}
{body}
{param_group}
[ \t]*       # any number of spaces and/or tabs
$            # end of line (since re.MULTILINE is passed)
""", re.MULTILINE | re.VERBOSE)

_re_hide = _mk_flag_re("hide?", 0,
    "Matches any line with #hide without any module name")

# Cell
def _get_unhidden_cells(cells):
    result = []
    for i,cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            if not _re_hide.findall(cell['source'].lower()) and cell['source'] != '': result.append(i)
    return result

def _read_nb(fname):
    "Read the notebook in `fname`."
    from pathlib import Path
    import nbformat
    with open(Path(fname),'r', encoding='utf8') as f: return nbformat.reads(f.read(), as_version=4)

# Cell
"""The code in this cell is a modified version of the one included in this repo: https://github.com/msm1089/ipynbname
# Copyright (c) 2020 Mark McPherson. """

# MIT License

# Copyright (c) 2020 Mark McPherson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import urllib.error
import urllib.request
from itertools import chain
from pathlib import Path, PurePath
from typing import Generator, Tuple, Union

import ipykernel
from jupyter_core.paths import jupyter_runtime_dir
from traitlets.config import MultipleInstanceError


FILE_ERROR = "Can't identify the notebook {}."
CONN_ERROR = "Unable to access server;\n" \
           + "ipynbname requires either no security or token based security."


def _list_maybe_running_servers(runtime_dir=None) -> Generator[dict, None, None]:
    """ Iterate over the server info files of running notebook servers.
    """
    if runtime_dir is None:
        runtime_dir = jupyter_runtime_dir()
    runtime_dir = Path(runtime_dir)

    if runtime_dir.is_dir():
        for file_name in chain(
            runtime_dir.glob('nbserver-*.json'),  # jupyter notebook (or lab 2)
            runtime_dir.glob('jpserver-*.json'),  # jupyterlab 3
        ):
            yield json.loads(file_name.read_bytes())


def _get_kernel_id() -> str:
    """ Returns the kernel ID of the ipykernel.
    """
    connection_file = Path(ipykernel.get_connection_file()).stem
    kernel_id = connection_file.split('-', 1)[1]
    return kernel_id


def _get_sessions(srv):
    """ Given a server, returns sessions, or HTTPError if access is denied.
        NOTE: Works only when either there is no security or there is token
        based security. An HTTPError is raised if unable to connect to a
        server.
    """
    try:
        qry_str = ""
        token = srv['token']
        if token:
            qry_str = f"?token={token}"
        url = f"{srv['url']}api/sessions{qry_str}"
        with urllib.request.urlopen(url) as req:
            return json.load(req)
    except Exception:
        raise urllib.error.HTTPError(CONN_ERROR)


def _find_nb() -> Union[Tuple[dict, PurePath], Tuple[None, None]]:
    try:
        kernel_id = _get_kernel_id()
    except (MultipleInstanceError, RuntimeError):
        return None, None  # Could not determine
    for srv in _list_maybe_running_servers():
        try:
            sessions = _get_sessions(srv)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return srv, PurePath(sess['notebook']['path'])
        except Exception:
            pass  # There may be stale entries in the runtime directory
    return None, None


def get_nb_name() -> str:
    """ Returns the short name of the notebook w/o the .ipynb extension,
        or raises a FileNotFoundError exception if it cannot be determined.
    """
    _, path = _find_nb()
    if path:
        return path.stem
    else:
        return

def get_colab_nb_name():
    import requests
    from urllib.parse import unquote
    from pathlib import Path
    d = requests.get('http://172.28.0.2:9000/api/sessions').json()[0]
    fname = unquote(d['name'])
    fid = unquote(d['path'].split('=')[1])
    if 'https://github.com' in fid: fname = fid
    else: fname = Path(f'drive/MyDrive/Colab Notebooks/{fname}')
    return fname

def get_nb_path() -> Path:
    """ Returns the absolute path of the notebook,
        or raises a FileNotFoundError exception if it cannot be determined.
    """
    if is_colab(): return get_colab_nb_name()
    else:
        srv, path = _find_nb()
        if srv and path:
            root_dir = Path(srv.get('root_dir') or srv['notebook_dir'])
            return root_dir / path
        else:
            return

def get_script_path(nb_name=None):
    if nb_name is None: nb_name = get_nb_path()
    return str(nb_name).replace(".ipynb", ".py")

# Cell
def is_lab():
    import re
    import psutil
    return any(re.search('jupyter-lab', x) for x in psutil.Process().parent().cmdline())

def is_colab():
    from IPython.core import getipython
    return 'google.colab' in str(getipython.get_ipython())

def to_local_time(t, time_format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(time_format, time.localtime(t))

def _save_nb(nb_name, attempts=5, wait=1, verbose=True):
    """
    Save and checkpoints current jupyter notebook. 1 attempt per second approx.
    """
    from IPython.core.display import Javascript, display, HTML
    import time
    try:
        saved = False
        current_time = time.time()
        if is_colab():
            if verbose: print(f'cannot save the notebook in Google Colab. Last saved {to_local_time(os.path.getmtime(nb_name))}.')
            time.sleep(wait)
        else:
            for i in range(attempts):
                if is_lab():
                    script = """
                    this.nextElementSibling.focus();
                    this.dispatchEvent(new KeyboardEvent('keydown', {key:'s', keyCode: 83, metaKey: true}));
                    """
                    display(HTML(('<img src onerror="{}" style="display:none">'
                                  '<input style="width:0;height:0;border:0">').format(script)))
                else:
                    display(Javascript('IPython.notebook.save_checkpoint();'))
                for j in range(10):
                    time.sleep(wait/10)
                    saved_time = os.path.getmtime(nb_name)
                    if  saved_time>= current_time: break
                if saved_time >= current_time:
                    saved = True
                    break
        if verbose:
            if saved: print(f'{nb_name} saved at {to_local_time(saved_time)}.')
            else: print(f"{nb_name} couldn't be saved.")
        time.sleep(wait)
    except:
        if verbose: print(f"{nb_name} couldn't be saved.")

def maybe_mount_drive():
    from pathlib import Path
    from google.colab.drive import mount
    if not Path("/content/drive").exists(): mount("/content/drive")

# Cell
from fastcore.script import *

@call_parse
def nb2py(nb:      Param("absolute or relative full path to the notebook you want to convert to a python script", str)=None,
          folder:  Param("absolute or relative path to folder of the script you will create. Defaults to current nb's directory", str)=None,
          name:    Param("name of the script you want to create. Defaults to current nb name .ipynb by .py", str)=None,
          run:     Param("import and run the script", store_true)=False,
          verbose: Param("controls verbosity", store_false)=True,
         ):
    "Converts a notebook to a python script in a predefined folder."

    import os
    from pathlib import Path
    from .imports import import_file_from_module
    try: import nbformat
    except ImportError: raise ImportError("You need to install nbformat to use nb2py!")

    # make sure drive is mounted when using Colab
    if is_colab(): maybe_mount_drive()

    # nb path & name
    if nb is not None:
        nb_path = Path(nb)
        nb_path = nb_path.parent/f"{nb_path.stem}.ipynb"
    else:
        try:
            nb_path = get_nb_path()
            if nb_path is None:
                print("nb2py couldn't get the nb_name. Pass it as an argument and re-run nb2py.")
                return
        except:
            print("nb2py couldn't get the nb_name. Pass it as an argument and re-run nb2py.")
            return
    nb_name = nb_path.name
    assert os.path.isfile(nb_path), f"nb2py couldn't find {nb_path}. Please, confirm the path is correct."

    # save nb: only those that are run from the notebook itself
    if not is_colab() and nb is None:
        _save_nb(nb_name, attempts=5, wait=1, verbose=True)

    # script path & name
    if folder is not None: folder = Path(folder)
    else: folder = nb_path.parent
    if name is not None: name = f"{Path(name).stem}.py"
    else: name = f"{nb_path.stem}.py"
    script_path = folder/name

    # delete file if exists and create script_path folder if doesn't exist
    if os.path.exists(script_path): os.remove(script_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    # Write script header
    with open(script_path, 'w') as f:
        f.write(f'# -*- coding: utf-8 -*-\n')
        f.write(f'"""{nb_name}\n\n')
        f.write(f'Automatically generated.\n\n')
        if nb_path is not None:
            f.write(f'Original file is located at:\n')
            f.write(f'    {nb_path}\n')
        f.write(f'"""')

    # identify convertible cells (excluding empty and those with hide flags)
    nb = _read_nb(nb_path)
    idxs = _get_unhidden_cells(nb['cells'])
    pnb = nbformat.from_dict(nb)
    pnb['cells'] = [pnb['cells'][i] for i in idxs]

    # clean up cells and write script
    sep = '\n'* 2
    for i,cell in enumerate(pnb['cells']):
        source_str = cell['source'].replace('\r', '')
        code_lines = source_str.split('\n')
        if code_lines == ['']: continue
        while code_lines[0] == '': code_lines = code_lines[1:]
        while code_lines[-1] == '': code_lines = code_lines[:-1]
        cl = []
        for j in range(len(code_lines)):
            if list(set(code_lines[j].split(" "))) == ['']:
                code_lines[j] = ''
            if i == 0 or code_lines[j-1] != '' or code_lines[j] != '':
                cl.append(code_lines[j])
        code_lines = cl
        code = sep + '\n'.join(code_lines)
        with open(script_path, 'a', encoding='utf8') as f: f.write(code)

    # check script exists
    assert os.path.isfile(script_path), f"an error occurred during the export and {script_path} doesn't exist"
    if verbose:
        print(f"{nb_name} converted to {script_path}")
    if run: import_file_from_module(script_path)
    else: return str(script_path)