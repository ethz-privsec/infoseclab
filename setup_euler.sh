module load gcc/8.2.0 python_gpu/3.10.4
echo "Creating virtual environment"
python -m virtualenv .venv --system-site-packages

echo "Activating environment"
source .venv/bin/activate

echo "Reinstalling Jupyter Lab inside of environment"
python -m pip install jupyterlab --force-reinstall

echo "Installing Jupyter extension from pip to connect to Colab"
python -m pip install jupyter_http_over_ws

echo "Patching Jupyter extension"
ORIGINAL_STRING="_AUTH_URL_QUERY_PARAM = 'jupyter_http_over_ws_auth_url'"
SUBSTITUTE_STRING="_AUTH_URL_QUERY_PARAM = '127.0.0.1:8888'"
FILE_TO_PATCH=".venv/lib64/python3.10/site-packages/jupyter_http_over_ws/handlers.py"
sed -i -e "s/${ORIGINAL_STRING}/${SUBSTITUTE_STRING}/g" $FILE_TO_PATCH

echo "Installing extension into Jupyter Lab"
.venv/bin/jupyter serverextension enable --py jupyter_http_over_ws

echo "Setup finished, deactivating environment"
deactivate