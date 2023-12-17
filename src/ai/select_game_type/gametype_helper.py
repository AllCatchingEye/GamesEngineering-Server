from state.gametypes import GameGroup, Gametype

GAME_TYPE_TO_GAME_GROUP_MAPPING = {
    GameGroup.HIGH_SOLO: [Gametype.SOLO],
    GameGroup.MID_SOLO: [Gametype.WENZ, Gametype.GEIER],
    GameGroup.LOW_SOLO: [Gametype.FARBWENZ, Gametype.FARBGEIER],
    GameGroup.SAUSPIEL: [Gametype.SAUSPIEL],
}


def game_type_to_game_group(game_type: Gametype):
    for game_group, game_types in GAME_TYPE_TO_GAME_GROUP_MAPPING.items():
        if game_type in game_types:
            return game_group
    raise KeyError("Game type " + game_type.name + " has no according game group")
