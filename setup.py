#!/usr/bin/env python
# encoding: utf-8
# TODO Requirements are not really minimal at the moment...
# TODO Having PyTest in requirements not optimal, fix to make travis use
#      the tests_require

import os
import sys

from setuptools import Command, find_packages, setup

authors = [u"Daniel Suess", u"Milan Holzäpfel"]
author_emails = ["daniel@dsuess.me", "mail@mjh.name"]
name = "mpnum"
description = "matrix product representation library"
if os.path.isfile('README.txt'):
    long_description = open('README.txt').read()
else:
    long_description = description
year = "2016"


try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/" + name)
    from mpnum import __version__ as version
except:
    version = "unknown"


class PyTest(Command):
    """Perform tests"""

    description = "Runs test suite"
    user_options = [
        ('selector=', None, "Specifies the tests to run"),
        ('covreport', None, "Run coverage report from tests"),
        ('pdb', None, "Whether to run pdb on failure"),
    ]

    def initialize_options(self):
        self.selector = 'not long'
        self.covreport = None
        self.pdb = False

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        args = [] if not self.pdb else ['--pdb']

        if self.covreport is None:
            # Run doctests afterwards because they auto-import things. We want
            # to see missing imports.
            errno = pytest.main(['-m', self.selector, 'tests'] + args)
            if errno != 0:
                raise SystemExit(errno)

            errno = pytest.main(['--doctest-modules', 'mpnum'] + args)
            if errno != 0:
                raise SystemExit(errno)

        else:
            errno = pytest.main(['-m', self.selector, '--cov-report', 'term',
                                 '--cov-report', 'html', '--cov=mpnum',
                                 '--doctest-modules', 'mpnum', 'tests'] + args)
            if errno != 0:
                raise SystemExit(errno)


_install_requires = [
    'SciPy>=0.15',
    'NumPy>=1.5.1',
    'six>=1.0',
    'PyTest>=2.8.7',
    'h5py>=2.4'
]


def _get_install_requires(req, not_on_rtd=['scipy', 'numpy', 'h5py']):
    """Remove packages which cannot be installed on readthedocs.org

    scipy and numpy are available on readthedocs as system packages,
    but they must not appear in install_requires (see
    http://docs.readthedocs.org/en/latest/faq.html).

    """
    on_rtd = os.environ.get('READTHEDOCS') == 'True'
    if on_rtd:
        req = [dep for dep in req
               if all(blocker.lower() not in dep.lower()
                      for blocker in not_on_rtd)]
    return req


if __name__ == '__main__':
    setup(
        name=name,
        author=', '.join(authors),
        author_email=', '.join(author_emails),
        url="https://github.com/dseuss/mpnum",
        version=version,
        packages=find_packages(exclude=['tests']),
        license="BSD",
        description=description,
        long_description=long_description,
        install_requires=_get_install_requires(_install_requires),
        package_dir={'mpnum': 'mpnum'},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Intended Audience :: Science/Research"
        ],
        platforms=['ALL'],
        cmdclass={
            'test': PyTest
        }
    )
