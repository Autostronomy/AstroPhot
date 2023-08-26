import os
root_path = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]

with open(os.path.join(root_path, "VERSION"), "r") as f:
    version = f.read().strip()

print(root_path)

print(version)

# Update CITATION.cff
######################################################################
with open(os.path.join(root_path, "CITATION.cff"), "r") as f:
    CITATION = f.readlines()

with open(os.path.join(root_path, "CITATION.cff"), "w") as f:
    for line in CITATION:
        if line.startswith("version:"):
            f.write(f"version: {version}\n")
        else:
            f.write(line)

# Update docs/conf.py
######################################################################
with open(os.path.join(root_path, "docs", "conf.py"), "r") as f:
    CONF = f.readlines()

with open(os.path.join(root_path, "docs", "conf.py"), "w") as f:
    for line in CONF:
        if line.startswith("version ="):
            f.write(f"version = \"{version[:version.rfind('.')]}\"\n")
        elif line.startswith("release ="):
            f.write(f"release = \"{version}\"\n")
        else:
            f.write(line)

# Update astrophot/__init__.py
######################################################################
with open(os.path.join(root_path, "astrophot", "__init__.py"), "r") as f:
    INIT = f.readlines()

with open(os.path.join(root_path, "astrophot", "__init__.py"), "w") as f:
    for line in INIT:
        if line.startswith("__version__ ="):
            f.write(f"__version__ = \"{version}\"\n")
        else:
            f.write(line)
