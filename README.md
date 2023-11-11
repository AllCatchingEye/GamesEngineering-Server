# schafkopf-server

Schafkopf Game Server

## Getting Started

### Prerequisites

- [Python >= 3.10](https://www.python.org/downloads/)

### Installing

<<<<<<< README.md
* Clone this repository
* (Optional but recommended) Create an virtual environment with `python -m venv venv`
  * Activate the virtual environment with `source venv/bin/activate`
    * On Windows use `venv\Scripts\activate.bat`
* Install the requirements with `pip install -r requirements.txt && pip install -r requirements-dev.txt`

### Format, Lint and Typecheck

- Windows: `.\scripts\lint.ps1`
- Linux/Mac: `./scripts/lint.sh`

### Testing

* Windows: `.\scripts\test.ps1`
* Linux/Mac: `./scripts/test.sh`

### FAQ

#### Exec tests: Permission denied

You may have to grant your user executable rights for executing the script file.

For Mac- and Linux-Systems, `chmod u+x` could be a solution. But make sure you know what you are doing!

#### Exec tests: No module named coverage

When I run the tests, I get this error: `/Users/<user>/anaconda3/bin/python: No module named coverage`

Reason: It may be the case that your shell uses anaconda's python environment which can lead to issues. 

How to fix? Run `conda deactivate` and try it again.

### Run sample server

1. Start server with command `python src/sample_server.py`
2. Start first client with command `python src/sample_client.py start`
3. You will receive a key for joining the game. This key can be given to other clients,
   so that they can join your running game
4. Start second client with command `python src/sample_client.py join <key>`. `<key>` is the join key,
   which the first client gets when starting a game, which you need to provide to enter the game
5. Both clients should now be in the game
