import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from utils.reward_functions import calculate_rewards  # Replace with the actual module name


class TestCalculateRewards(unittest.TestCase):
    """
    Tests for the calculate_rewards function in utils/reward_functions.py

    This class tests the edge cases for the calculate_rewards function which
    include:

    - New game
    - Game over
    - Life lost
    - No life lost
    - Gravity timer exceeds threshold
    - Gravity timer does not exceed threshold
    - Lines cleared reward
    - Penalty scaling
    - Reward scaling
    - Empty board no penalties
    - Total rewards and penalties bounded
    - Combined penalties and rewards
    - Missing stats keys

    """

    def test_new_game(self):
        """
        Test that when it is a new game, the reward is 0.
        """
        current_stats = {}
        prev_stats = {}
        lines_cleared = 0
        game_over = False
        action_history = []
        result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
        self.assertEqual(result["Total_Reward"], 0)

    def test_game_over(self):
        """
        Test that when the game is over, the penalty is -1.
        """
        current_stats = {"lives_left": 0}
        prev_stats = {"lives_left": 1}
        lines_cleared = 0
        game_over = True
        action_history = []
        result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
        self.assertEqual(result["game_over_penalty"], -1)

    # TODO refactor test_life_lost, test_no_life_lost, and test_gravity_timer_exceeds_threshold to
    def test_life_lost(self):
        """
        Test that when a life is lost, the penalty is -0.9.
        """
        current_stats = {"lives_left": 0}
        prev_stats = {"lives_left": 1}
        lines_cleared = 0
        game_over = False
        action_history = []
        result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
        self.assertEqual(result["lost_a_life"], -0.9)

    def test_no_life_lost(self):
        """
        Test that when no life is lost, the result contains the scaled penalties and rewards.
        """
        current_stats = {"lives_left": 1}
        prev_stats = {"lives_left": 1}
        lines_cleared = 0
        game_over = False
        action_history = []
        result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
        self.assertIsNotNone(result["scaled_penalties"])
        self.assertIsNotNone(result["scaled_rewards"])

    def test_gravity_timer_exceeds_threshold(self):
        """
        Test that when the gravity timer exceeds the threshold, the penalty is negative.
        """
        current_stats = {"gravity_timer": 30, "gravity_interval": 60}
        prev_stats = {"gravity_timer": 0}
        lines_cleared = 0
        game_over = False
        action_history = []
        result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
        self.assertLess(result["scaled_penalties"]["gravity_timer"], 0)

    # def test_gravity_timer_does_not_exceed_threshold(self):
    #     """
    #     Test that when the gravity timer does not exceed the threshold, the penalty is 0.
    #     """
    #     current_stats = {"gravity_timer": 20, "gravity_interval": 60}
    #     prev_stats = {"gravity_timer": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertEqual(result["scaled_penalties"]["gravity_timer"], 0)

    # def test_lines_cleared_reward(self):
    #     """
    #     Test that when lines are cleared, the reward is positive.
    #     """
    #     current_stats = {}
    #     prev_stats = {}
    #     lines_cleared = 2
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertGreater(result["scaled_rewards"]["lines_cleared_reward"], 0)

    # def test_penalty_scaling(self):
    #     """
    #     Test that when the penalty is scaled, the result is negative.
    #     """
    #     current_stats = {"max_height": 10, "holes": 5}
    #     prev_stats = {"max_height": 5, "holes": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertLessEqual(result["scaled_penalties"]["height_penalty"], -0.9)
    #     self.assertLessEqual(result["scaled_penalties"]["hole_penalty"], -0.9)

    # def test_reward_scaling(self):
    #     """
    #     Test that when the reward is scaled, the result is positive.
    #     """
    #     current_stats = {}
    #     prev_stats = {}
    #     lines_cleared = 4
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertLessEqual(result["scaled_rewards"]["lines_cleared_reward"], 1)

    # def test_empty_board_no_penalties(self):
    #     """
    #     Test that when the board is empty, there are no penalties.
    #     """
    #     current_stats = {"max_height": 0, "holes": 0}
    #     prev_stats = {"max_height": 0, "holes": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertEqual(result["total_penalties"], 0)
    #     self.assertEqual(result["total_rewards"], 0)

    # def test_total_rewards_and_penalties_bounded(self):
    #     """
    #     Test that the total rewards and penalties are bounded between -1 and 1.
    #     """
    #     current_stats = {"max_height": 15, "holes": 50}
    #     prev_stats = {"max_height": 10, "holes": 5}
    #     lines_cleared = 4
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     total = result["total_rewards"] + result["total_penalties"]
    #     self.assertGreaterEqual(total, -1)
    #     self.assertLessEqual(total, 1)

    # def test_combined_penalties_and_rewards(self):
    #     """
    #     Test that the combined penalties and rewards are as expected.
    #     """
    #     current_stats = {"max_height": 12, "holes": 4, "gravity_timer": 40, "gravity_interval": 60}
    #     prev_stats = {"max_height": 5, "holes": 2, "gravity_timer": 20}
    #     lines_cleared = 3
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertLess(result["scaled_penalties"]["height_penalty"], 0)
    #     self.assertLess(result["scaled_penalties"]["hole_penalty"], 0)
    #     self.assertLess(result["scaled_penalties"]["gravity_timer"], 0)
    #     self.assertGreater(result["scaled_rewards"]["lines_cleared_reward"], 0)

    # def test_missing_stats_keys(self):
    #     """
    #     Test that when there are missing stats keys, the result contains the scaled penalties and rewards.
    #     """
    #     current_stats = {"max_height": 10}  # Missing "holes"
    #     prev_stats = {"max_height": 5, "holes": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertIn("scaled_penalties", result)  # Ensure it still computes scaled penalties

    # def test_gravity_timer_at_threshold(self):
    #     """
    #     Test that when the gravity timer is exactly at the threshold, the penalty is 0.
    #     """
    #     current_stats = {"gravity_timer": 30, "gravity_interval": 60}
    #     prev_stats = {"gravity_timer": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertEqual(result["scaled_penalties"]["gravity_timer"], 0)

    # def test_penalty_boundaries(self):
    #     """
    #     Test that penalties do not exceed the defined boundaries.
    #     """
    #     current_stats = {"max_height": 15, "holes": 50}
    #     prev_stats = {"max_height": 10, "holes": 5}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertGreaterEqual(result["scaled_penalties"]["height_penalty"], -0.0525)
    #     self.assertLessEqual(result["scaled_penalties"]["hole_penalty"], -0.02)

    # def test_rescaling_on_exceeding_boundaries(self):
    #     """
    #     Test that when the sum of penalties and rewards exceeds the bounds, it is rescaled to [-1, 1].
    #     """
    #     current_stats = {"max_height": 20, "holes": 200}
    #     prev_stats = {"max_height": 10, "holes": 100}
    #     lines_cleared = 10  # High reward case
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     total = result["total_rewards"] + result["total_penalties"]
    #     self.assertGreaterEqual(total, -1)
    #     self.assertLessEqual(total, 1)

    # def test_missing_keys_defaults(self):
    #     """
    #     Test that missing keys default to 0 penalties or rewards.
    #     """
    #     current_stats = {"max_height": 10}  # Missing "holes"
    #     prev_stats = {"max_height": 5}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     result = calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)
    #     self.assertEqual(result["scaled_penalties"].get("hole_penalty", 0), 0)

    # def test_invalid_input(self):
    #     """
    #     Test that invalid inputs (e.g., negative max_height) are handled properly.
    #     """
    #     current_stats = {"max_height": -10, "holes": 5}
    #     prev_stats = {"max_height": 5, "holes": 0}
    #     lines_cleared = 0
    #     game_over = False
    #     action_history = []
    #     with self.assertRaises(ValueError):  # Assuming you handle invalid values with an exception
    #         calculate_rewards(current_stats, prev_stats, lines_cleared, game_over, action_history)


if __name__ == "__main__":
    unittest.main()
