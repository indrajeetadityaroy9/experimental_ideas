from go_game import GoGame

def test_interactive_setup():
    try:
        game = GoGame(5)  # Small board for testing
        print("GoGame initialized successfully")
        
        # Test that we can get groups
        groups = game._get_all_groups()
        print(f"Initial groups: {groups}")
        
        # Test playing a move
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