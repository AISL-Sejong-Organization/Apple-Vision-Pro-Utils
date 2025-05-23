from setuptools import find_packages, setup

package_name = 'visionpro_controller'

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hyeonsu",
    maintainer_email="hans324oh@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    # setup.py 예시
    entry_points={
        "console_scripts": [
            "visionpro_manipulator_ctrl_node = visionpro_controller.manipulator_controller:main",
        ],
    },
)
