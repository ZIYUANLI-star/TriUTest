conda run -n MuTAP-master python -c "import sys; print(sys.executable)"
conda run -n MuTAP-master python - <<'PY'
import sysconfig, os
print(os.path.join(sysconfig.get_paths()['scripts'], 'mut.py'))
PY
