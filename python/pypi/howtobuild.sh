# Copy the files in the pypi folder to a temporary directory
mkdir /tmp/pypi
cp * /tmp/pypi

# Change to temporary folder
cd /tmp/pypi

# Clone specific tag of astra toolbox to temporary folder
git clone --branch master --depth 1 https://www.github.com/astra-toolbox/astra-toolbox

mkdir -p astra/plugins
touch astra/__init__.py
touch astra/plugins/__init__.py

# Build/upload PyPI package
python setup.py sdist upload -r pypi
