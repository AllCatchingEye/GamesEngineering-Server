Write-Output "[SORTIN IMPORTS]"
isort .
Write-Output "[FORMAT CODE]"
black .
Write-Output "[LINTING]"
pylint src
Write-Output "[TYPE CHECKING]"
mypy .