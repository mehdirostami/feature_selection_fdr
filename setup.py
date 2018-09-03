from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
      
setup(name='feature_selection_fdr',
      version='0.1',
      description='Feature selection with FDR control.',
      long_description=readme(),
      classifiers=[
	'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'      ],
      url='https://github.com/mehdirostami/feature_selection_fdr',
      author='Mehdi Rostami',
      author_email='mehdi.rostamiforooshani@mail.utoronto.ca',
      packages=['feature_selection_fdr'],
      install_requires=[
          'numpy', 'matplotlib', 'cvxopt', 'sklearn', 'multiprocessing',
	'statsmodels', 'pandas', 'glmnet', 'scipy'          
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
zip_safe=False)
