import nbformat
from nbconvert import PythonExporter

with open("", "r", encoding="utf-8") as f:
    code = f.read()
    
exporter = PythonExporter()
body, resources = exporter.from_filename("")
nb = nbformat.reads(body, as_version=4)

with open("your_notebook.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
