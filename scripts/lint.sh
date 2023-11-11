source venv/bin/activate

echo "[SORTIN IMPORTS]"
isort .
echo "[FORMAT CODE]"
black .
echo "[UNUSED CODE]"
vulture src
echo "[LINTING]"
pylint src
echo "[TYPE CHECKING]"
mypy --exclude '.*_test\.py$' .