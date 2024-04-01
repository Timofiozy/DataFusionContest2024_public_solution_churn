import shutil
from distutils.core import Distribution, Extension
from pathlib import Path

from Cython.Build import build_ext, cythonize
import numpy

cython_dir = Path("data_fusion_contest_2024_churn/ext")
extension = Extension(
    "data_fusion_contest_2024_churn.ext.objectives",
    [
        str(cython_dir / "objectives.pyx"),
    ],
    extra_compile_args=["-O3", "-std=c99"],
    include_dirs=[numpy.get_include()]
)

ext_modules = cythonize(
    [extension],
    include_path=[cython_dir],
    language_level=3,
    annotate=True
)
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = Path(output).relative_to(cmd.build_lib)
    shutil.copyfile(output, relative_extension)
