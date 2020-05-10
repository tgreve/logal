from distutils.core import setup

setup(
    name='logal',
    version='0.1dev',
    author='Thomas R. Greve',
    author_email='t.r.greve@gmail.com',
    package_dir={'logal': 'src'},
    packages=['logal'],
    long_description=open('README.txt').read(),
)
