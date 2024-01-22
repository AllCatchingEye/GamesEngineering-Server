# paths are:
# .\src\ai\select_game_type\two_layer_nn\models\binary_classifier.pth -> ai\select_game_type\two_layer_nn\models
# .\src\ai\select_game_type\two_layer_nn\models\game_classifier.pth -> ai\select_game_type\two_layer_nn\models

# include all paths under .\src\ai\select_card\models\dql\iteration_02\params\custom\* -> ai\select_card\models\dql\iteration_02\params\custom

# pyinstaller one file, open a console for logging, set name to "schafkopf-server.exe"
pyinstaller --onefile --add-data=".\src\ai\select_game_type\two_layer_nn\models\binary_classifier.pth;ai\select_game_type\two_layer_nn\models" --add-data=".\src\ai\select_game_type\two_layer_nn\models\game_classifier.pth;ai\select_game_type\two_layer_nn\models" --add-data=".\src\ai\select_card\models\dql\iteration_02\params\custom\*;ai\select_card\models\dql\iteration_02\params\custom" --console --name schafkopf-server.exe .\src\server.py
