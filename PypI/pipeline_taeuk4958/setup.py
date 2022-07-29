import os
from setuptools import setup, find_packages

# python setup.py bdist_wheel
# twine upload dist/pipeline_taeuk4958-1.0.7-py3-none-any.whl

version_file = os.path.join(os.getcwd(), 'pipeline_taeuk4958', 'version.py')             # version이 명시된 file

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(
        name='pipeline_taeuk4958',      # package 명
        version=get_version(),
        description='package to load file from docker local in docker container during container running',
        author='taeuk4958 ',
        author_email='taeuk4958@gmail.com',
        url='https://github.com/HibernationNo1/project_4_kubeflow_pipeline.git',
        packages=find_packages(),
        include_package_data=True,
        license='Apache License 2.0',
        install_requires=['numpy', 'addict'])
    
    