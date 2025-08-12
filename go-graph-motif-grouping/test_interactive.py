import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from go_game import GoGame

def test_interactive_setup():
    try:
        game = GoGame(5)
        print("GoGame initialized successfully")
        
        groups = game._get_all_groups()
        print(f"Initial groups: {groups}")
        
        result = game.play_move(2, 2)
        print(f"Move result: {result}")
        
        groups = game._get_all_groups()
        print(f"Groups after move: {groups}")
        
        print("All tests passed!")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_interactive_setup()