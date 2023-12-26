import unittest

from ai.select_game_type.gametype_helper import game_type_to_game_group
from state.gametypes import GameGroup, Gametype


class TestClass(unittest.TestCase):
    def test_game_type_to_game_group(self):
        self.assertEqual(game_type_to_game_group(Gametype.SOLO), GameGroup.HIGH_SOLO)
        self.assertEqual(game_type_to_game_group(Gametype.WENZ), GameGroup.MID_SOLO)
        self.assertEqual(game_type_to_game_group(Gametype.GEIER), GameGroup.MID_SOLO)
        self.assertEqual(game_type_to_game_group(Gametype.FARBWENZ), GameGroup.LOW_SOLO)
        self.assertEqual(
            game_type_to_game_group(Gametype.FARBGEIER), GameGroup.LOW_SOLO
        )
        self.assertEqual(game_type_to_game_group(Gametype.SAUSPIEL), GameGroup.SAUSPIEL)
