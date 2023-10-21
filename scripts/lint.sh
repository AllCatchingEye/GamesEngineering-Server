echo "[SORTIN IMPORTS]"
isort .
echo "[FORMAT CODE]"
black .
echo "[LINTING]"
pylint src
echo "[TYPE CHECKING]"
mypy --exclude '.*_test\.py$' .