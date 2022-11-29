module load gcc/8.2.0 python_gpu/3.10.4
echo "Creating virtual environment"
python -m virtualenv .venv --system-site-packages

echo "Activating environment"
source .venv/bin/activate

echo "Reinstalling Jupyter Lab inside of environment"
python -m pip install jupyterlab --force-reinstall

echo "Installing Jupyter extension to use Colab"
python -m pip install jupyter_http_over_ws
.venv/bin/jupyter serverextension enable --py jupyter_http_over_ws

echo "Deactivating"
deactivate