from os.path import abspath, dirname, join

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

about: dict = {}
here = abspath(dirname(__file__))
with open(join(here, "play3d", "__about__.py")) as f:
    exec(f.read(), about)

setuptools.setup(
    name="play3d",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    license=about["__license__"],
    packages=['play3d'],
    install_requires=[line.strip() for line in open(join(here, "requirements.txt")).readlines()],
    description="Basic 3D world playground with animations and completely from scratch.",
    long_description=open(join(dirname(__file__), "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords="python 3d play3d three_d projection mvp",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)