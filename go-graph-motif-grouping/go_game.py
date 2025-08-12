import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sys
import os
from collections import defaultdict

class GoGame:
    def __init__(self, size=9):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)] 
        self.captured_stones = []  
        self.move_history = [] 
        self.current_player = 1  
        self.subgraph_history = []  
        self.motif_history = []  # Track motif formations
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
        self.motif_edge_collections = []
        
        self.dihedral_transforms = [
            lambda w: w,  # Identity
            lambda w: (w[1], -w[0]),  # 90 rotation
            lambda w: (-w[0], -w[1]),  # 180 rotation
            lambda w: (-w[1], w[0]),  # 270° rotation
            lambda w: (w[0], -w[1]),  # Reflection across x-axis
            lambda w: (-w[0], w[1]),  # Reflection across y-axis
            lambda w: (w[1], w[0]),   # Reflection across main diagonal
            lambda w: (-w[1], -w[0])  # Reflection across anti-diagonal
        ]
    
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
    
    def is_atari(self, group):
        return len(self.get_liberties(group)) == 1
    
    def is_eye(self, x, y, color):
        if self.board[x][y] != 0:
            return False
            
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx][ny] != color:
                return False
                
        return True
    
    def has_two_eyes(self, group):
        color = self.board[next(iter(group))[0]][next(iter(group))[1]]
        eyes = []
        
        for x, y in self.get_liberties(group):
            if self.is_eye(x, y, color):
                eyes.append((x, y))
                
        if len(eyes) < 2:
            return False
            
        return len(eyes) >= 2
    
    def _match_template(self, template, anchor_x, anchor_y, color_swap=1):
        min_x = min(w[0] for w in template)
        max_x = max(w[0] for w in template)
        min_y = min(w[1] for w in template)
        max_y = max(w[1] for w in template)
        
        if anchor_x + min_x < 0 or anchor_x + max_x >= self.size or anchor_y + min_y < 0 or anchor_y + max_y >= self.size:
            return False
            
        for transform in self.dihedral_transforms:
            match = True
            for w in template:
                tw = transform(w)
                bx, by = anchor_x + tw[0], anchor_y + tw[1]
                
                if not (0 <= bx < self.size and 0 <= by < self.size):
                    match = False
                    break

                required = template[w]
                actual = self.board[bx][by]
                
                if required == '*':
                    continue
                    
                if required in [-1, 1]:
                    required = required * color_swap
                    
                if required != actual:
                    match = False
                    break
                    
            if match:
                return True
                
        return False
    
    def _detect_atari_motifs(self):
        motifs = []
        groups = self._get_all_groups()
        
        for group in groups:
            if self.is_atari(group):
                color = self.board[next(iter(group))[0]][next(iter(group))[1]]
                motifs.append(("atari", group, color))
                
        return motifs
    
    def _detect_eye_motifs(self):
        motifs = []
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    if self.is_eye(i, j, 1):
                        motifs.append(("eye", {(i, j)}, 1))
                    elif self.is_eye(i, j, 2):
                        motifs.append(("eye", {(i, j)}, 2))
                        
        return motifs
    
    def _detect_life_motifs(self):
        motifs = []
        groups = self._get_all_groups()
        
        for group in groups:
            if self.has_two_eyes(group):
                color = self.board[next(iter(group))[0]][next(iter(group))[1]]
                motifs.append(("life", group, color))
                
        return motifs
    
    def _detect_capture_motifs(self):
        motifs = []
        groups = self._get_all_groups()
        
        for group in groups:
            # Check if group is in atari
            if self.is_atari(group):
                color = self.board[next(iter(group))[0]][next(iter(group))[1]]
                opponent = 3 - color
                motifs.append(("capture", group, opponent))
                
        return motifs
    
    def _detect_connection_motifs(self):
        motifs = []
        groups = self._get_all_groups()
        
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1, g2 = groups[i], groups[j]
                
                if self.board[next(iter(g1))[0]][next(iter(g1))[1]] != self.board[next(iter(g2))[0]][next(iter(g2))[1]]:
                    continue
                    
                liberties1 = self.get_liberties(g1)
                liberties2 = self.get_liberties(g2)
                
                if liberties1 & liberties2:
                    motifs.append(("connection", (g1, g2)))
                    
        return motifs
    
    def _find_motifs(self):
        motifs = []
        
        motifs.extend(self._detect_atari_motifs())
        motifs.extend(self._detect_eye_motifs())
        motifs.extend(self._detect_life_motifs())
        motifs.extend(self._detect_capture_motifs())
        motifs.extend(self._detect_connection_motifs())
                
        return motifs
    
    def is_valid_move(self, x, y, color=None):
        if color is None:
            color = self.current_player
            
        if self.board[x][y] != 0:
            return False
            
        original_board = [row[:] for row in self.board]
        self.board[x][y] = color
        
        placed_group = self.get_group(x, y)
        if self.get_liberties(placed_group):
            self.board = original_board
            return True
            
        captured = False
        opponent = 3 - color
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx][ny] == opponent:
                group = self.get_group(nx, ny)
                if not self.get_liberties(group):
                    captured = True
                    break

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
            
        self.board[x][y] = color
        self.move_history.append((x, y, color))
        
        opponent = 3 - color
        captured_groups = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx][ny] == opponent:
                group = self.get_group(nx, ny)
                if not self.get_liberties(group):
                    captured_groups.append(group)
        
        for group in captured_groups:
            self.capture_group(group)
        
        motifs = self._find_motifs()
        if motifs:
            self.motif_history.append({
                'move': (x, y, color),
                'motifs': motifs
            })
        
        self.current_player = 3 - self.current_player
        return True
    
    def undo_move(self):
        if not self.move_history:
            return False
            
        x, y, color = self.move_history.pop()
        self.board[x][y] = 0
        self.current_player = color
        
        if self.subgraph_history:
            self.subgraph_history.pop()
            
        if self.motif_history:
            self.motif_history.pop()
            
        return True
    
    def _setup_visualization(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            plt.subplots_adjust(right=0.8)
            
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
            self.fig.canvas.mpl_connect('button_press_event', self._on_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _update_visualization(self):
        node_colors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1: 
                    node_colors.append('black')
                elif self.board[i][j] == 2: 
                    node_colors.append('white')
                else:  
                    node_colors.append('lightblue')
        
        self.node_collection.set_facecolors(node_colors)

        if self.group_edge_collections:
            for collection in self.group_edge_collections:
                if collection:
                    try:
                        collection.remove()
                    except (AttributeError, ValueError):
                        pass
            self.group_edge_collections = []
        
        if self.motif_edge_collections:
            for collection in self.motif_edge_collections:
                if collection:
                    try:
                        if hasattr(collection, 'remove'):
                            collection.remove()
                        elif isinstance(collection, list):
                            for item in collection:
                                if hasattr(item, 'remove'):
                                    item.remove()
                    except (AttributeError, ValueError):
                        pass
            self.motif_edge_collections = []
        
        groups = self._get_all_groups()
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        if not hasattr(self, 'subgraph_history'):
            self.subgraph_history = []
        
        current_groups = [sorted(list(group)) for group in groups]
        if not self.subgraph_history or self.subgraph_history[-1] != current_groups:
            self.subgraph_history.append(current_groups)
        
        for i, (group, color) in enumerate(zip(groups, group_colors)):
            if len(group) > 1:  
                subgraph = self.graph.subgraph(group)
                edge_collection = nx.draw_networkx_edges(
                    subgraph, self.pos, edge_color=[color], width=2, alpha=0.8, ax=self.ax
                )
                if edge_collection is not None:
                    self.group_edge_collections.append(edge_collection)
        
        motifs = self._find_motifs()
        motif_styles = {
            'eye': {'color': 'gold', 'style': 'solid', 'width': 3},
            'connection': {'color': 'purple', 'style': 'solid', 'width': 3}, 
            'atari': {'color': 'red', 'style': 'solid', 'width': 3},
            'life': {'color': 'green', 'style': 'solid', 'width': 3},
            'capture': {'color': 'orange', 'style': 'solid', 'width': 3}
        }
        
        for motif in motifs:
            motif_type = motif[0]
            style = motif_styles.get(motif_type, {'color': 'cyan', 'style': 'solid', 'width': 3})
            
            if motif_type in ['eye', 'atari', 'life']:
                _, positions, _ = motif
                if len(positions) > 0:
                    subgraph = self.graph.subgraph(positions)
                    edge_collection = nx.draw_networkx_edges(
                        subgraph, self.pos, 
                        edge_color=style['color'], 
                        width=style['width'], 
                        alpha=0.9, 
                        style=style['style'],
                        ax=self.ax
                    )
                    if edge_collection is not None:
                        self.motif_edge_collections.append(edge_collection)
                        
                    if motif_type == 'eye':
                        pos_list = list(positions)
                        if len(pos_list) == 1:
                            pos = pos_list[0]
                            if pos in self.pos:
                                x, y = self.pos[pos]
                                circle = Circle((x, y), 0.2, color=style['color'], 
                                               alpha=0.7, fill=True)
                                self.ax.add_patch(circle)
                                self.motif_edge_collections.append(circle)
            elif motif_type == "connection":
                _, (group1, group2) = motif
                for pos1 in group1:
                    for pos2 in group2:
                        if pos1 in self.pos and pos2 in self.pos:
                            x_coords = [self.pos[pos1][0], self.pos[pos2][0]]
                            y_coords = [self.pos[pos1][1], self.pos[pos2][1]]
                            line = self.ax.plot(x_coords, y_coords, 
                                              color=style['color'], 
                                              linewidth=style['width'], 
                                              alpha=0.9, 
                                              linestyle=style['style'])
                            self.motif_edge_collections.append(line[0])
            elif motif_type == "capture":
                _, group, _ = motif
                if group:
                    subgraph = self.graph.subgraph(group)
                    edge_collection = nx.draw_networkx_edges(
                        subgraph, self.pos, 
                        edge_color=style['color'], 
                        width=style['width'], 
                        alpha=0.9, 
                        style=style['style'],
                        ax=self.ax
                    )
                    if edge_collection is not None:
                        self.motif_edge_collections.append(edge_collection)
        
        self.ax.set_title(f"Go Game Board ({self.size}x{self.size}) - Player {'Black' if self.current_player == 1 else 'White'} to move")
        
        if self.captured_stones:
            captured_patch = mpatches.Patch(color='red', label='Captured Stones (Highlighted)')
            if len(self.legend_handles) == 3:
                self.legend_handles.append(captured_patch)
            plt.legend(handles=self.legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            if len(self.legend_handles) > 3:
                self.legend_handles = self.legend_handles[:3]
            plt.legend(handles=self.legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if self.captured_edge_collection:
            try:
                self.captured_edge_collection.remove()
            except (AttributeError, ValueError):
                pass
            self.captured_edge_collection = None
            
        if self.captured_stones:
            valid_captured_stones = [stone for stone in self.captured_stones if stone in self.pos]
            if valid_captured_stones:
                for stone in valid_captured_stones:
                    x, y = self.pos[stone]
                    circle = Circle((x, y), 0.3, color='red', alpha=0.7, fill=False, linewidth=3)
                    self.ax.add_patch(circle)
                    self.motif_edge_collections.append(circle)
        
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
    
    def _on_key_press(self, event):
        if event.key == 'u' or event.key == 'U':
            if self.undo_move():
                self._update_visualization()
                print("Move undone")
        elif event.key == 'r' or event.key == 'R':
            self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
            self.captured_stones = []
            self.move_history = []
            self.current_player = 1
            if hasattr(self, 'subgraph_history'):
                self.subgraph_history = []
            if hasattr(self, 'motif_history'):
                self.motif_history = []
            self._update_visualization()
            print("Game reset")
        elif event.key == 'h' or event.key == 'H':
            self.print_subgraph_history()
        elif event.key == 'm' or event.key == 'M':
            self.print_motif_history()
        elif event.key == 'q' or event.key == 'Q':
            plt.close(self.fig)
            sys.exit(0)
    
    def visualize(self, highlight_captured=False, save_as=None):
        pos = {(i, j): (j, -i) for i in range(self.size) for j in range(self.size)}
        
        node_colors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1: 
                    node_colors.append('black')
                elif self.board[i][j] == 2:
                    node_colors.append('white')
                else:
                    node_colors.append('lightblue')
        
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(self.graph, pos, edge_color='black', alpha=0.3)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, edgecolors='black', linewidths=1)
        
        if highlight_captured and self.captured_stones:
            valid_captured_stones = [stone for stone in self.captured_stones if stone in pos]
            
            if valid_captured_stones:
                for stone in valid_captured_stones:
                    x, y = pos[stone]
                    circle = Circle((x, y), 0.3, color='red', alpha=0.7, fill=False, linewidth=3)
                    plt.gca().add_patch(circle)
        
        labels = {(i, j): f"{i},{j}" for i in range(self.size) for j in range(self.size)}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=6)
        
        plt.title(f"Go Game Board ({self.size}x{self.size}) - Player {'Black' if self.current_player == 1 else 'White'} to move")
        plt.axis('equal')
        plt.axis('off')
        
        black_patch = mpatches.Patch(color='black', label='Black Stones')
        white_patch = mpatches.Patch(color='white', edgecolor='black', label='White Stones')
        empty_patch = mpatches.Patch(color='lightblue', label='Empty Intersections')
        plt.legend(handles=[black_patch, white_patch, empty_patch], loc='upper right')
        
        if highlight_captured and self.captured_stones:
            captured_patch = mpatches.Patch(color='red', label='Captured Stones (Highlighted)')
            plt.legend(handles=[black_patch, white_patch, empty_patch, captured_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {save_as}")
        else:
            plt.show()
    
    def print_subgraph_history(self):
        if not hasattr(self, 'subgraph_history') or not self.subgraph_history:
            print("No subgraph history available.")
            return
            
        print("\nSubgraph Formation History:")
        for i, groups in enumerate(self.subgraph_history):
            print(f"Step {i}:")
            for j, group in enumerate(groups):
                print(f"  Group {j}: {group}")
            print()
    
    def print_motif_history(self):
        if not self.motif_history:
            print("No motif history available.")
            return
            
        print("\nMotif Formation History:")
        for i, entry in enumerate(self.motif_history):
            move = entry['move']
            motifs = entry['motifs']
            print(f"Move {i+1} at {move}:")
            for motif in motifs:
                motif_type = motif[0]
                if motif_type == "eye":
                    _, positions, color = motif
                    color_name = "Black" if color == 1 else "White"
                    print(f"  Eye formed by {color_name} stones at positions: {list(positions)}")
                elif motif_type == "atari":
                    _, group, color = motif
                    color_name = "Black" if color == 1 else "White"
                    print(f"  Atari for {color_name} group: {list(group)}")
                elif motif_type == "life":
                    _, group, color = motif
                    color_name = "Black" if color == 1 else "White"
                    print(f"  Life for {color_name} group: {list(group)}")
                elif motif_type == "capture":
                    _, group, color = motif
                    color_name = "Black" if color == 1 else "White"
                    print(f"  Capture opportunity for {color_name} on group: {list(group)}")
                elif motif_type == "connection":
                    _, (group1, group2) = motif
                    print(f"  Connection between groups: {list(group1)} and {list(group2)}")
            print()

    def start_interactive_game(self):
        self._setup_visualization()
        self._update_visualization()
        plt.show()

def demo():
    game = GoGame(9)
    
    game.play_move(4, 4)
    game.play_move(3, 3) 
    game.play_move(5, 5)  
    game.play_move(2, 2) 
    
    print("Go board after 4 moves:")
    game.visualize(save_as="go_board_after_4_moves.png")
    
    game.play_move(3, 4)  
    game.play_move(3, 5)  
    game.play_move(4, 5)  
    game.play_move(2, 4)  
    
    game.print_subgraph_history()
    game.print_motif_history()
    
    print("\nGo board with captured stones highlighted:")
    game.visualize(highlight_captured=True, save_as="go_board_with_captured.png")

def interactive_demo():
    game = GoGame(9)
    game._setup_visualization()
    game._update_visualization()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        demo()