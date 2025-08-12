import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sys
import os

class GoGame:
    def __init__(self, size=9):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)] 
        self.captured_stones = []  
        self.move_history = [] 
        self.current_player = 1  
        self.subgraph_history = []  
        self.graph = nx.grid_2d_graph(size, size)
        
        for i in range(size):
            for j in range(size):
                if i > 0 and j > 0:
                    self.graph.add_edge((i, j), (i-1, j-1))
                if i > 0 and j < size - 1:
                    self.graph.add_edge((i, j), (i-1, j+1))
                if i < size - 1 and j > 0:
                    self.graph.add_edge((i, j), (i+1, j-1))
                if i < size - 1 and j < size - 1:
                    self.graph.add_edge((i, j), (i+1, j+1))
                    
        self.fig = None
        self.ax = None
        self.pos = None
        self.edge_collection = None
        self.node_collection = None
        self.label_collection = None
        self.legend_handles = []
        self.captured_edge_collection = None
        self.group_edge_collections = []
    
    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors
    
    def get_group(self, x, y):
        if self.board[x][y] == 0:
            return set()
        
        color = self.board[x][y]
        group = set()
        to_check = [(x, y)]
        visited = set()
        
        while to_check:
            cx, cy = to_check.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            if self.board[cx][cy] == color:
                group.add((cx, cy))
                for nx, ny in self.get_neighbors(cx, cy):
                    if (nx, ny) not in visited:
                        to_check.append((nx, ny))
        
        return group
    
    def get_liberties(self, group):
        liberties = set()
        for x, y in group:
            for nx, ny in self.get_neighbors(x, y):
                if self.board[nx][ny] == 0:
                    liberties.add((nx, ny))
        return liberties
    
    def is_valid_move(self, x, y, color=None):
        if color is None:
            color = self.current_player
            
        # Check if position is empty
        if self.board[x][y] != 0:
            return False
            
        # Temporarily place the stone
        original_board = [row[:] for row in self.board]
        self.board[x][y] = color
        
        # Check if the placed stone has liberties
        placed_group = self.get_group(x, y)
        if self.get_liberties(placed_group):
            self.board = original_board
            return True
            
        # Check if any opponent group is captured
        captured = False
        opponent = 3 - color
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx][ny] == opponent:
                group = self.get_group(nx, ny)
                if not self.get_liberties(group):
                    captured = True
                    break
        
        # Restore board
        self.board = original_board
        return captured or self.get_liberties(placed_group)
    
    def capture_group(self, group):
        for x, y in group:
            self.board[x][y] = 0
        self.captured_stones.extend(list(group))
    
    def play_move(self, x, y, color=None):
        if color is None:
            color = self.current_player
            
        if not self.is_valid_move(x, y, color):
            return False
            
        # Place the stone
        self.board[x][y] = color
        self.move_history.append((x, y, color))
        
        # Check for captures
        opponent = 3 - color
        captured_groups = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx][ny] == opponent:
                group = self.get_group(nx, ny)
                if not self.get_liberties(group):
                    captured_groups.append(group)
        
        # Capture opponent groups
        for group in captured_groups:
            self.capture_group(group)
        
        # Switch player
        self.current_player = 3 - self.current_player
        return True
    
    def undo_move(self):
        if not self.move_history:
            return False
            
        x, y, color = self.move_history.pop()
        self.board[x][y] = 0
        self.current_player = color
        
        # Remove last entry from subgraph history
        if self.subgraph_history:
            self.subgraph_history.pop()
            
        return True
    
    def _setup_visualization(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            plt.subplots_adjust(bottom=0.2, right=0.8)
            
            self.pos = {(i, j): (j, -i) for i in range(self.size) for j in range(self.size)}
            
            self.edge_collection = nx.draw_networkx_edges(
                self.graph, self.pos, edge_color='black', alpha=0.3, ax=self.ax
            )
            
            node_colors = ['lightblue'] * (self.size * self.size)
            self.node_collection = nx.draw_networkx_nodes(
                self.graph, self.pos, node_color=node_colors,
                node_size=500, edgecolors='black', linewidths=1, ax=self.ax
            )
            
            labels = {(i, j): f"{i},{j}" for i in range(self.size) for j in range(self.size)}
            self.label_collection = nx.draw_networkx_labels(
                self.graph, self.pos, labels, font_size=6, ax=self.ax
            )
            
            black_patch = mpatches.Patch(color='black', label='Black Stones')
            white_patch = mpatches.Patch(color='white', edgecolor='black', label='White Stones')
            empty_patch = mpatches.Patch(color='lightblue', label='Empty Intersections')
            self.legend_handles = [black_patch, white_patch, empty_patch]
            
            ax_undo = plt.axes([0.1, 0.05, 0.1, 0.075])
            self.btn_undo = Button(ax_undo, 'Undo')
            self.btn_undo.on_clicked(self._on_undo_clicked)
            
            ax_reset = plt.axes([0.25, 0.05, 0.1, 0.075])
            self.btn_reset = Button(ax_reset, 'Reset')
            self.btn_reset.on_clicked(self._on_reset_clicked)
            
            ax_exit = plt.axes([0.4, 0.05, 0.1, 0.075])
            self.btn_exit = Button(ax_exit, 'Exit')
            self.btn_exit.on_clicked(self._on_exit_clicked)
            
            # Connect click event
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _update_visualization(self):
        node_colors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:  # Black stone
                    node_colors.append('black')
                elif self.board[i][j] == 2:  # White stone
                    node_colors.append('white')
                else:  # Empty intersection
                    node_colors.append('lightblue')
        
        self.node_collection.set_facecolors(node_colors)
        
        # Remove previous group edges
        for collection in self.group_edge_collections:
            collection.remove()
        self.group_edge_collections = []
        
        # Draw group connections
        groups = self._get_all_groups()
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        # Track subgraph formation history
        if not hasattr(self, 'subgraph_history'):
            self.subgraph_history = []
        
        # Create a representation of current groups for history tracking
        current_groups = [sorted(list(group)) for group in groups]
        if not self.subgraph_history or self.subgraph_history[-1] != current_groups:
            self.subgraph_history.append(current_groups)
        
        for i, (group, color) in enumerate(zip(groups, colors)):
            if len(group) > 1:  # Only draw edges for groups with more than one stone
                subgraph = self.graph.subgraph(group)
                edge_collection = nx.draw_networkx_edges(
                    subgraph, self.pos, edge_color=[color], width=3, alpha=0.7, ax=self.ax
                )
                if edge_collection is not None:
                    self.group_edge_collections.append(edge_collection)
        
        # Update title
        self.ax.set_title(f"Go Game Board ({self.size}x{self.size}) - Player {'Black' if self.current_player == 1 else 'White'} to move")
        
        # Update legend with captured stones if needed
        if self.captured_stones:
            captured_patch = mpatches.Patch(color='red', label='Captured Stones (Highlighted)')
            if len(self.legend_handles) == 3:
                self.legend_handles.append(captured_patch)
            plt.legend(handles=self.legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            if len(self.legend_handles) > 3:
                self.legend_handles = self.legend_handles[:3]
            plt.legend(handles=self.legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Highlight captured stones
        if self.captured_edge_collection:
            self.captured_edge_collection.remove()
            self.captured_edge_collection = None
            
        if self.captured_stones:
            valid_captured_stones = [stone for stone in self.captured_stones if stone in self.pos]
            if valid_captured_stones:
                captured_subgraph = self.graph.subgraph(valid_captured_stones)
                self.captured_edge_collection = nx.draw_networkx_edges(
                    captured_subgraph, self.pos, edge_color='red', width=3, style='dashed', ax=self.ax
                )
        
        self.fig.canvas.draw()
    
    def _get_all_groups(self):
        groups = []
        visited = set()
        
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in visited and self.board[i][j] != 0:
                    group = self.get_group(i, j)
                    groups.append(group)
                    visited.update(group)
        
        return groups
    
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        x, y = int(round(-event.ydata)), int(round(event.xdata))
        
        if 0 <= x < self.size and 0 <= y < self.size:
            success = self.play_move(x, y)
            if success:
                self._update_visualization()
            else:
                print(f"Invalid move at ({x}, {y})")
    
    def _on_undo_clicked(self, event):
        if self.undo_move():
            self._update_visualization()
    
    def _on_reset_clicked(self, event):
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.captured_stones = []
        self.move_history = []
        self.current_player = 1
        if hasattr(self, 'subgraph_history'):
            self.subgraph_history = []
        self._update_visualization()
    
    def _on_exit_clicked(self, event):
        plt.close(self.fig)
        sys.exit(0)
    
    def visualize(self, highlight_captured=False, save_as=None):
        # Create position mapping for visualization
        pos = {(i, j): (j, -i) for i in range(self.size) for j in range(self.size)}
        
        # Create node colors
        node_colors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:  # Black stone
                    node_colors.append('black')
                elif self.board[i][j] == 2:  # White stone
                    node_colors.append('white')
                else:  # Empty intersection
                    node_colors.append('lightblue')
        
        # Draw the graph
        plt.figure(figsize=(10, 10))
        
        # Draw edges (board grid lines)
        nx.draw_networkx_edges(self.graph, pos, edge_color='black', alpha=0.3)
        
        # Draw nodes with original colors
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                               node_size=500, edgecolors='black', linewidths=1)
        
        # Highlight captured stones if requested
        if highlight_captured and self.captured_stones:
            # Create a list of captured stone positions that are valid
            valid_captured_stones = [stone for stone in self.captured_stones if stone in pos]
            
            # Create positions and colors for captured stones
            if valid_captured_stones:
                captured_pos = {stone: pos[stone] for stone in valid_captured_stones}
                # Draw captured stones with a red outline to indicate capture
                nx.draw_networkx_nodes(self.graph, captured_pos, 
                                       node_color='none', node_size=500, 
                                       edgecolors='red', linewidths=3)
        
        # Add labels for coordinates
        labels = {(i, j): f"{i},{j}" for i in range(self.size) for j in range(self.size)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=6)
        
        # Customize plot
        plt.title(f"Go Game Board ({self.size}x{self.size}) - Player {'Black' if self.current_player == 1 else 'White'} to move")
        plt.axis('equal')
        plt.axis('off')
        
        # Create legend
        black_patch = mpatches.Patch(color='black', label='Black Stones')
        white_patch = mpatches.Patch(color='white', edgecolor='black', label='White Stones')
        empty_patch = mpatches.Patch(color='lightblue', label='Empty Intersections')
        plt.legend(handles=[black_patch, white_patch, empty_patch], loc='upper right')
        
        if highlight_captured and self.captured_stones:
            captured_patch = mpatches.Patch(color='red', label='Captured Stones (Highlighted)')
            plt.legend(handles=[black_patch, white_patch, empty_patch, captured_patch], loc='upper right')
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {save_as}")
        else:
            plt.show()
    
    def print_subgraph_history(self):
        """Print the history of subgraph formations."""
        if not hasattr(self, 'subgraph_history') or not self.subgraph_history:
            print("No subgraph history available.")
            return
            
        print("\nSubgraph Formation History:")
        for i, groups in enumerate(self.subgraph_history):
            print(f"Step {i}:")
            for j, group in enumerate(groups):
                print(f"  Group {j}: {group}")
            print()

    def start_interactive_game(self):
        self._setup_visualization()
        self._update_visualization()
        plt.show()

def demo():
    game = GoGame(9)
    
    # Play some moves to demonstrate
    game.play_move(4, 4)  # Black stone at center
    game.play_move(3, 3)  # White stone
    game.play_move(5, 5)  # Black stone
    game.play_move(2, 2)  # White stone
    
    print("Go board after 4 moves:")
    game.visualize(save_as="go_board_after_4_moves.png")
    
    game.play_move(3, 4)  # Black
    game.play_move(3, 5)  # White
    game.play_move(4, 5)  # Black
    game.play_move(2, 4)  # White
    
    game.print_subgraph_history()
    
    print("\nGo board with captured stones highlighted:")
    game.visualize(highlight_captured=True, save_as="go_board_with_captured.png")

def interactive_demo():
    game = GoGame(9)
    
    def _on_history_clicked(event):
        game.print_subgraph_history()
    
    game._setup_visualization()
    
    ax_history = plt.axes([0.55, 0.05, 0.1, 0.075])
    btn_history = Button(ax_history, 'History')
    btn_history.on_clicked(_on_history_clicked)
    
    game._update_visualization()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        demo()