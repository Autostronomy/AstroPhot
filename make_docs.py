import astrophot as ap
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell
import pkgutil
from types import ModuleType, FunctionType
import os
from textwrap import dedent
from inspect import cleandoc, getmodule, signature

skip_methods = [
    "to_valid",
    "topological_ordering",
    "to_static",
    "to_dynamic",
    "unlink",
    "update_graph",
    "save_state",
    "load_state",
    "append_state",
    "link",
    "graphviz",
    "graph_print",
    "graph_dict",
    "from_valid",
    "fill_params",
    "fill_kwargs",
    "fill_dynamic_values",
    "clear_params",
    "build_params_list",
    "build_params_dict",
    "build_params_array",
]


def dot_path(path):
    i = path.rfind("AstroPhot")
    path = path[i + 10 :]
    path = path.replace("/", ".")
    return path[:-3]


def gather_docs(module, module_only=False):
    docs = {}
    for name in module.__all__:
        obj = getattr(module, name)
        if module_only and not isinstance(obj, ModuleType):
            continue
        if isinstance(obj, type):
            if obj.__doc__ is None:
                continue
            docs[name] = cleandoc(obj.__doc__)
            subfuncs = [docs[name]]
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                if attr in skip_methods:
                    continue
                attrobj = getattr(obj, attr)
                if not isinstance(attrobj, FunctionType):
                    continue
                if attrobj.__doc__ is None:
                    continue
                sig = str(signature(attrobj)).replace("self,", "").replace("self", "")
                subfuncs.append(f"<u>**method:**</u> {attr}{sig}\n\n" + cleandoc(attrobj.__doc__))
            if len(subfuncs) > 1:
                docs[name] = "\n\n".join(subfuncs)
        elif isinstance(obj, FunctionType):
            if obj.__doc__ is None:
                continue
            sig = str(signature(obj))
            docs[name] = "<u>**signature:**</u> " + name + sig + "\n\n" + cleandoc(obj.__doc__)
        elif isinstance(obj, ModuleType):
            docs[name] = gather_docs(obj)
        else:
            print(f"!!!unexpected type {type(obj)}!!!")
    return docs


def make_cells(mod_dict, path, depth=2):
    print(mod_dict.keys())
    cells = []
    for k in mod_dict:
        if isinstance(mod_dict[k], str):
            cells.append(new_markdown_cell(f"{'#'*depth} {path}.{k}\n\n" + mod_dict[k]))
        elif isinstance(mod_dict[k], dict):
            print(k)
            cells += make_cells(mod_dict[k], path=path + "." + k, depth=depth + 1)
    return cells


output_dir = "docs/source/astrophotdocs"
all_ap = gather_docs(ap, True)

for submodule in all_ap:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(f"# {submodule}")] + make_cells(
        all_ap[submodule], f"astrophot.{submodule}"
    )

    filename = f"{submodule}.ipynb"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
