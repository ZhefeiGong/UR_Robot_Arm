## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    version='0.0.0',
    packages=['ur5e_ctrl_jeff'],
    package_dir={'': 'scripts'},
    # requires=['rospy']
)

setup(**setup_args)
