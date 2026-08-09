
set -e
for f in tests/test_*.py; do
    mod="tests.$(basename "$f" .py)"
    python -c "
import importlib
m = importlib.import_module('$mod')
for name in dir(m):
    if name.startswith('test_'):
        getattr(m, name)()
print('$mod: OK')
"
done
