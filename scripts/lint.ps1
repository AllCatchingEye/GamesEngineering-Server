Write-Output "[SORTIN IMPORTS]"
isort .
Write-Output "[FORMAT CODE]"
black .
Write-Output "[UNUSED CODE]"
vulture src
Write-Output "[LINTING]"
pylint src
Write-Output "[TYPE CHECKING]"
mypy --exclude '.*_test\.py$' .