import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from tqdm import tqdm
from agents import Player, RandomAgent, GreedyAgent
import inspect
from helpers import *

player_colors = ['red', 'blue', 'orange', 'white', 'green', 'yellow', 'purple', 'cyan', 'magenta']

resource_colors = {
    "sheep": 'lightgreen',
    "wood": 'saddlebrown',
    "wheat": 'gold',
    "brick": 'indianred',
    "ore": 'gray',
    "desert": 'khaki'
}

@dataclass
class Hex:
    """Represents a hexagonal tile on the board"""
    id: int
    resource: str
    roll_number: int
    position: Tuple[float, float]
    adjacent_hexes: Set[int] = field(default_factory=set)
    adjacent_vertices: Set[int] = field(default_factory=set)
    adjacent_edges: Set[int] = field(default_factory=set)
    has_robber: bool = False

@dataclass
class Vertex:
    """Represents a vertex (intersection point) where settlements/cities can be built"""
    id: int
    position: Tuple[float, float]
    adjacent_vertices: Set[int] = field(default_factory=set)
    adjacent_hexes: Set[int] = field(default_factory=set)
    adjacent_edges: Set[int] = field(default_factory=set)
    owner: Optional[int] = None
    is_settlement: bool = False
    is_city: bool = False
    port: Optional[str] = None

@dataclass
class Edge:
    """Represents an edge where roads can be built"""
    id: int
    vertex1: int
    vertex2: int
    adjacent_hexes: Set[int] = field(default_factory=set)
    adjacent_edges: Set[int] = field(default_factory=set)
    owner: Optional[int] = None

@dataclass
class GameConfig:
    """Configuration parameters for the Catan game"""
    num_players: int = 4
    n_die: int = 2
    die_sides: int = 6
    max_turns: int = 100
    victory_points_to_win: int = 10
    randomized_desert: bool = False
    num_deserts: int = 1
    allow_trading: bool = True
    allow_bank_trading: bool = True
    allow_dev_card_reuse: dict = field(default_factory=lambda: {
        "knight": True,
        "victory_point": False,
        "road_building": False,
        "year_of_plenty": False,
        "monopoly": False
    })

    verbose: int = 2  # 0: silent, 1: summary, 2: detailed, 3: debug
    def to_string(self):
        return (f"{self.num_players}p_{self.victory_points_to_win}vp_"
            f"{self.max_turns}maxt_"
            f"{int(self.allow_trading)}t_"
            f"{int(self.allow_bank_trading)}bt")

class CatanGame:
    """Game class that handles Catan logic and the game state. Although probably unnecessary for this
    implementation and use, the player classes are designed to query potential actions through this
    object, so all game logic is handled here.
    
    For any event involving player choice, the player has some methodology for selecting an action (if any, as end_turn
    counts), querying the game to ensure validity and execute that action, and re-query the player for further specification.
    For example, a player may choose to build a settlement, and then must query the game for valid settlement locations and then
    choose one. Another example would be deciding to play a dev card, then choosing which card to play, and then if it's year of
    plenty or monopoly, choosing resources accordingly. For RL implementation later, these multi-step actions are simplified
    into recursive single-step actions.
    """
    def __init__(self, config: GameConfig = None, seed: int = None, players=None):
        self.config = config or GameConfig()
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.hexes: Dict[int, Hex] = {}
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Edge] = {}
        self.ports: Dict[int, Dict] = {}
        self.players: List[Player] = []
        
        self.current_player: int = 0
        self.turn: int = 0
        self.robber_hex_id: Optional[int] = None
        
        # inventories
        self.inventories: Dict[int, Dict[str, int]] = {}
        self.player_dev_cards: Dict[int, Dict[str, int]] = {}
        self.knights_played: Dict[int, int] = {}
        
        # can't play dev cards bought in the same turn
        self.dev_cards_bought_this_turn: Dict[int, List[str]] = {}
        self.dev_card_played_this_turn: Dict[int, bool] = {}
        
        # achievements
        self.longest_road_info: Tuple[Optional[int], int] = (None, 0)
        self.largest_army_info: Tuple[Optional[int], int] = (None, 0)
        
        # longest road cache
        self._longest_road_cache: Dict[int, int] = {}
        self._road_network_dirty: Set[int] = set()
        
        # verbosity
        self.verbose: int = self.config.verbose or 0
        
        # building
        self.building_costs = {
            "road": {"wood": 1, "brick": 1},
            "settlement": {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1},
            "city": {"wheat": 2, "ore": 3},
            "development_card": {"sheep": 1, "wheat": 1, "ore": 1}
        }
        
        # Development card deck, known to all players for card distributions
        development_cards = {
            "knight": 14,
            "victory_point": 5,
            "road_building": 2,
            "year_of_plenty": 2,
            "monopoly": 2
        }

        self.dev_card_deck = []
        for card_type, count in development_cards.items():
            self.dev_card_deck.extend([card_type] * count)
        random.shuffle(self.dev_card_deck)
        
        self.build_board()
        self.init_players(players)
    
    def _player_display(self, player_id):
        """Convert player_id to 1-indexed display format"""
        return player_id + 1
    
    def build_board(self):
        """Build a standard Catan board with proper hexagonal topology"""

        hex_layout = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, -1), (1, 0), (1, 1), (1, 2)],
            [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2)],
            [(3, -2), (3, -1), (3, 0), (3, 1)],
            [(4, -2), (4, -1), (4, 0)]
        ]
        
         # Standard Catan resource distribution
        """"resource_deck = [
            t
            for t in resource_colors.keys()
            if t != "desert"
            for _ in range(19)
        ]
        random.shuffle(resource_deck)

        #resources = []
        for _ in range(19 - self.config.num_deserts): # generalize to board size
            resources.append(resource_deck.pop())"""

        # would be fun to generalize to different board sizes, shapes, and distributions of initial resources later
        resources = [ # standard resource distribution
            "wheat", "wheat", "wheat", "wheat",
            "wood", "wood", "wood", "wood",
            "brick", "brick", "brick",
            "sheep", "sheep", "sheep", "sheep",
            "ore", "ore", "ore"
        ]

        roll_numbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12] # standard roll numbers

        random.shuffle(resources)
        random.shuffle(roll_numbers)

        if self.config.randomized_desert:
            for _ in range(self.config.num_deserts):
                insert_idx = random.randint(0, len(resources))
                resources.insert(insert_idx, "desert")
        else:
            for _ in range(self.config.num_deserts):
                resources.insert(len(resources)//2, "desert")

        # Create hexes
        hex_id = 0
        axial_to_hex_id = {}
        roll_idx = 0

        for row_idx, row in enumerate(hex_layout):
            for col_idx, (q, r) in enumerate(row):
                x = r + 0.5 * q
                y = -q * np.sqrt(3)/2
                
                resource = resources[hex_id]
                roll_num = 7 if resource == "desert" else roll_numbers[roll_idx]
                if resource != "desert":
                    roll_idx += 1
                
                hex_obj = Hex(
                    id=hex_id,
                    resource=resource,
                    roll_number=roll_num,
                    position=(x, y),
                    has_robber=(resource == "desert")
                )
                
                self.hexes[hex_id] = hex_obj
                axial_to_hex_id[(q, r)] = hex_id
                
                if resource == "desert":
                    self.robber_hex_id = hex_id
                
                hex_id += 1

        def get_neighbors_offset(q, r):
            if q % 2 == 0:
                return [(q, r+1), (q-1, r), (q-1, r-1), (q, r-1), (q+1, r-1), (q+1, r)]
            else:
                return [(q, r+1), (q-1, r+1), (q-1, r), (q, r-1), (q+1, r), (q+1, r+1)]

        for (q, r), hex_id in axial_to_hex_id.items():
            neighbors = get_neighbors_offset(q, r)
            for neighbor_coords in neighbors:
                if neighbor_coords in axial_to_hex_id:
                    neighbor_id = axial_to_hex_id[neighbor_coords]
                    self.hexes[hex_id].adjacent_hexes.add(neighbor_id)

        vertex_id = 0
        vertex_positions = {}

        for hex_id, hex_obj in self.hexes.items():
            x, y = hex_obj.position
            
            hex_vertices = []
            for i in range(6):
                angle = np.pi / 3 * i + np.pi / 6
                vx = x + np.sqrt(3)/3 * np.cos(angle) 
                vy = y + np.sqrt(3)/3 * np.sin(angle)
                
                pos_key = (round(vx, 3), round(vy, 3))
                
                if pos_key not in vertex_positions:
                    vertex = Vertex(id=vertex_id, position=pos_key)
                    self.vertices[vertex_id] = vertex
                    vertex_positions[pos_key] = vertex_id
                    vertex_id += 1
                
                vid = vertex_positions[pos_key]
                hex_vertices.append(vid)
                self.vertices[vid].adjacent_hexes.add(hex_id)
            
            self.hexes[hex_id].adjacent_vertices = set(hex_vertices)

        vertex_adjacency_pairs = set()
        
        for hex_id, hex_obj in self.hexes.items():
            x, y = hex_obj.position
            
            vertex_angles = []
            for vid in hex_obj.adjacent_vertices:
                vx, vy = self.vertices[vid].position
                angle = np.arctan2(vy - y, vx - x)
                vertex_angles.append((angle, vid))
            
            vertex_angles.sort()
            sorted_vertices = [vid for _, vid in vertex_angles]
            
            for i in range(len(sorted_vertices)):
                v1 = sorted_vertices[i]
                v2 = sorted_vertices[(i + 1) % len(sorted_vertices)]
                edge_pair = tuple(sorted([v1, v2]))
                vertex_adjacency_pairs.add(edge_pair)
        
        for v1, v2 in vertex_adjacency_pairs:
            self.vertices[v1].adjacent_vertices.add(v2)
            self.vertices[v2].adjacent_vertices.add(v1)


        edge_id = 0
        
        for v1, v2 in vertex_adjacency_pairs:
            edge = Edge(id=edge_id, vertex1=v1, vertex2=v2)
            
            hex_set1 = self.vertices[v1].adjacent_hexes
            hex_set2 = self.vertices[v2].adjacent_hexes
            edge.adjacent_hexes = hex_set1.intersection(hex_set2)
            
            self.edges[edge_id] = edge
            
            self.vertices[v1].adjacent_edges.add(edge_id)
            self.vertices[v2].adjacent_edges.add(edge_id)
            
            for hid in edge.adjacent_hexes:
                self.hexes[hid].adjacent_edges.add(edge_id)
            
            edge_id += 1

        for edge_id, edge in self.edges.items():
            v1, v2 = edge.vertex1, edge.vertex2
            adjacent_edge_ids = (self.vertices[v1].adjacent_edges | 
                               self.vertices[v2].adjacent_edges)
            adjacent_edge_ids.discard(edge_id)
            edge.adjacent_edges = adjacent_edge_ids
        
        self._initialize_ports()
        
        if self.verbose >= 4:
            print(f"Board created: {len(self.hexes)} hexes, {len(self.vertices)} vertices, {len(self.edges)} edges, {len(self.ports)} ports")

    def _initialize_ports(self):
        """Initialize ports on the board edges according to standard Catan rules"""
        # Find edges on the perimeter of the board (edges adjacent to only 1 hex)
        perimeter_edges = []
        for edge_id, edge in self.edges.items():
            if len(edge.adjacent_hexes) == 1:
                perimeter_edges.append(edge_id)
        
        # Sort perimeter edges by angle from center for consistent placement
        center_x = sum(h.position[0] for h in self.hexes.values()) / len(self.hexes)
        center_y = sum(h.position[1] for h in self.hexes.values()) / len(self.hexes)
        
        def edge_angle(edge_id):
            v1_pos = self.vertices[self.edges[edge_id].vertex1].position
            v2_pos = self.vertices[self.edges[edge_id].vertex2].position
            mid_x = (v1_pos[0] + v2_pos[0]) / 2
            mid_y = (v1_pos[1] + v2_pos[1]) / 2
            return np.arctan2(mid_y - center_y, mid_x - center_x)
        
        perimeter_edges.sort(key=edge_angle)
        
        # 4 generic (3:1) and 5 specialized (2:1) port types
        port_types = ['3:1', '3:1', '3:1', '3:1',
                      'wheat', 'wood', 'brick', 'sheep', 'ore']
        random.shuffle(port_types)
        
        # place ports: typically every ~4th perimeter edge to get 9 ports
        num_ports = 9
        port_spacing = len(perimeter_edges) // num_ports
        
        for i in range(num_ports):
            edge_idx = i * port_spacing
            if edge_idx < len(perimeter_edges):
                edge_id = perimeter_edges[edge_idx]
                port_type = port_types[i] if i < len(port_types) else '3:1'
                
                self.ports[edge_id] = {
                    'type': port_type,
                    'ratio': '2:1' if port_type not in ['3:1'] else '3:1',
                    'resource': None if port_type == '3:1' else port_type
                }
    
    def init_players(self, players=None):
        """Initialize player data"""

        if players is None:
            players = [RandomAgent] * self.config.num_players
            print("No players passed, defaulting to RandomAgent for all players.")

        for i in range(self.config.num_players): # 1-index player IDs
            if inspect.isclass(players[i]):
                player_class = players[i] if i < len(players) else RandomAgent
                if player_class not in {RandomAgent, GreedyAgent}:
                    player = player_class(id=i, color=player_colors[i], config=self.config, network=None)
                else:
                    player = player_class(id=i, color=player_colors[i])
            else:
                player = players[i]
                player.victory_points = 0
                player.development_cards = defaultdict(int)
                player.knights_played = 0
                player.owned_roads = set()
                player.owned_vertices = set()
            self.players.append(player)
            
            self.inventories[i] = {
                'wheat': 0, 'wood': 0, 'brick': 0, 'sheep': 0, 'ore': 0
            }
            
            self.player_dev_cards[i] = {
                'knight': 0, 'victory_point': 0, 'road_building': 0,
                'year_of_plenty': 0, 'monopoly': 0
            }
            
            self.knights_played[i] = 0
            self.dev_cards_bought_this_turn[i] = []
            self.dev_card_played_this_turn[i] = False
    
    def setup_initial_placements(self):
        """Setup initial settlements and roads for all players"""
        if self.verbose >= 3:
            print("\n=== Initial Placement Phase ===")
        
        # First round
        for player_id in range(self.config.num_players):
            self.players[player_id].setup_initial_placements(self, round_num=1)
        
        # Second round (reverse order)
        for player_id in reversed(range(self.config.num_players)):
            self.players[player_id].setup_initial_placements(self, round_num=2)
    
    def can_afford(self, player_id, building_type):
        """Check if player can afford a building"""
        cost = self.building_costs[building_type]
        inventory = self.inventories[player_id]
        
        for resource, amount in cost.items():
            if inventory.get(resource, 0) < amount:
                return False
        return True
    
    def get_player_inventory(self, player_id):
        """Get a copy of player's inventory"""
        return dict(self.inventories[player_id])
    
    def get_valid_city_locations(self, player_id):
        """Get all valid locations where player can build a city"""
        valid = []
        for vertex_id, vertex in self.vertices.items():
            if vertex.owner == player_id and not vertex.is_city and vertex.is_settlement:
                valid.append(vertex_id)
        return valid
    
    def get_valid_settlement_locations(self, player_id, initial_placement=False):
        """Get all valid locations where player can build a settlement"""
        if not initial_placement:
            # Check settlement limit
            settlements = sum(1 for v in self.vertices.values() 
                            if v.is_settlement and v.owner == player_id)
            if settlements >= 5:
                return []
            
            # Must be connected to player's road
            owned_edges = [eid for eid, e in self.edges.items() if e.owner == player_id]
            candidates = set()
            for eid in owned_edges:
                edge = self.edges[eid]
                candidates.add(edge.vertex1)
                candidates.add(edge.vertex2)
        else:
            # Initial placement: any unowned vertex
            candidates = [vid for vid, v in self.vertices.items() if v.owner is None]
        
        # Filter with 2-distance rule
        valid = []
        for vid in candidates:
            vertex = self.vertices[vid]
            if vertex.owner is not None:
                continue
            if all(self.vertices[adj_vid].owner is None 
                   for adj_vid in vertex.adjacent_vertices):
                valid.append(vid)
        
        return valid
    
    def get_valid_road_locations(self, player_id, initial_placement=False, specific_vertex=None):
        """Get all valid locations where player can build a road
        Args:
            player_id: The player building the road
            initial_placement: If True, only return edges adjacent to most recent settlement
            specific_vertex: If provided, only return edges adjacent to this specific vertex
        """
        if specific_vertex is not None:
            # Return edges adjacent to a specific vertex (used during initial placement)
            return [eid for eid in self.vertices[specific_vertex].adjacent_edges 
                    if self.edges[eid].owner is None]
        
        # Get owned vertices and edges
        owned_vertices = [vid for vid, v in self.vertices.items() if v.owner == player_id]
        owned_edges = [eid for eid, e in self.edges.items() if e.owner == player_id]
        
        buildable = set()
        
        # Can build from any neighbors of owned vertex and edge that we have
        for vid in owned_vertices:
            buildable.update(self.vertices[vid].adjacent_edges)

        for eid in owned_edges:
            buildable.update(self.edges[eid].adjacent_edges)
        
        return [eid for eid in buildable if self.edges[eid].owner is None]
    
    def get_valid_robber_locations(self):
        # return all locations without a robber
        return [hid for hid in self.hexes.keys() if hid != self.robber_hex_id]
    
    def get_all_vertices(self):
        # get information about all vertices
        return {
            vid: {
                'owner': v.owner,
                'is_settlement': v.owner is not None and not v.is_city,
                'is_city': v.is_city,
                'adjacent_hexes': v.adjacent_hexes,
                'adjacent_vertices': v.adjacent_vertices
            }
            for vid, v in self.vertices.items()
        }
    
    def get_hex_info(self, hex_id):
        # get information about a hex
        hex_obj = self.hexes[hex_id]
        return {
            'resource': hex_obj.resource,
            'roll_number': hex_obj.roll_number,
            'has_robber': hex_obj.has_robber
        }
    
    def get_player_score(self, player_id):
        # calculate total victory points for a player
        settlements = sum(1 for v in self.vertices.values() 
                         if v.owner == player_id and not v.is_city)
        cities = sum(1 for v in self.vertices.values() 
                    if v.owner == player_id and v.is_city)
        
        score = settlements + 2 * cities + self.player_dev_cards[player_id]['victory_point']
        
        if self.longest_road_info[0] == player_id:
            score += 2
        if self.largest_army_info[0] == player_id:
            score += 2
        
        return score
    
    def get_player_dev_cards(self, player_id):
        # get a copy of player's development cards
        return dict(self.player_dev_cards[player_id])
    
    def get_player_trade_ratios(self, player_id):
        # Get the trade ratios available to a player based on their ports
        ratios = {}  # key: resource, value: best ratio available
        
        for edge_id, port_info in self.ports.items():
            edge = self.edges[edge_id]
            v1_owned = self.vertices[edge.vertex1].owner == player_id
            v2_owned = self.vertices[edge.vertex2].owner == player_id
            
            if v1_owned or v2_owned:
                if port_info['resource'] is None:
                    # Apply 3:1 to all resources that don't have a better ratio
                    for resource in {'wheat', 'wood', 'brick', 'sheep', 'ore'}:
                        if resource not in ratios:
                            ratios[resource] = 3
                else:
                    resource = port_info['resource']
                    ratios[resource] = 2
        if self.config.allow_bank_trading:
            # All resources default to 4:1 if not improved by ports
            for resource in {'wheat', 'wood', 'brick', 'sheep', 'ore'}:
                if resource not in ratios:
                    ratios[resource] = 4
        return ratios

    # execute player requests
    
    def build_city(self, player_id, vertex_id):
        if vertex_id not in self.get_valid_city_locations(player_id):
            return False
        
        # Pay cost
        if not self._pay_cost(player_id, "city"):
            return False
        
        # Execute
        self.vertices[vertex_id].is_city = True
        self.players[player_id].victory_points += 1
        
        if self.verbose >= 1:
            print(f"  🏛️ Player {self._player_display(player_id)} built a city at vertex {vertex_id} (+1 VP)")
        
        return True
    
    def build_settlement(self, player_id, vertex_id, initial_placement=False, second_placement=False):
        if vertex_id not in self.get_valid_settlement_locations(player_id, initial_placement):
            return False
        
        # Pay cost (not for initial placement)
        if not initial_placement and not self._pay_cost(player_id, "settlement"):
            return False
        
        self.vertices[vertex_id].owner = player_id
        self.vertices[vertex_id].is_settlement = True
        self.players[player_id].owned_vertices.add(vertex_id)
        self.players[player_id].victory_points += 1
        
        if self.verbose >= 1:
            print(f"  🏠 Player {self._player_display(player_id)} built a settlement at vertex {vertex_id} (+1 VP)")
        
        # Give resources for second initial settlement
        if second_placement:
            for hex_id in self.vertices[vertex_id].adjacent_hexes:
                hex_obj = self.hexes[hex_id]
                if hex_obj.resource != "desert":
                    self.inventories[player_id][hex_obj.resource] += 1
        
        return True

    def build_road(self, player_id, edge_id, initial_placement=False):
        """Build a road at the specified location"""
        if edge_id not in self.get_valid_road_locations(player_id, initial_placement):
            return False
        
        if not initial_placement and not self._pay_cost(player_id, "road"):
            return False
        
        self.edges[edge_id].owner = player_id
        self.players[player_id].owned_roads.add(edge_id)
        
        # mark player's road network as needing recalculation
        self._road_network_dirty.add(player_id)
        
        # update elongest road
        if not initial_placement:
            if self.verbose >= 3:
                print(f"  🛤️ Player {self._player_display(player_id)} built a road on edge {edge_id}")
            self._update_longest_road()
        
        return True
    
    def buy_development_card(self, player_id):
        # buy a dev card
        if not self.dev_card_deck:
            #if self.allow_dev_card_reuse:
            return False
        
        if not self._pay_cost(player_id, "development_card"):
            return False
        
        card = self.dev_card_deck.pop()
        self.player_dev_cards[player_id][card] += 1
        
        # track cards bought this turn
        self.dev_cards_bought_this_turn[player_id].append(card)
        
        if self.verbose >= 3:
            print(f"  📜 Player {self._player_display(player_id)} bought a {card} card")
        
        if card == "victory_point":
            self.players[player_id].victory_points += 1
            if self.verbose >= 1:
                print(f"  📜 Player {self._player_display(player_id)} gained a victory point card (+1 VP)")
        
        return True

    def play_development_card(self, player_id, card_type):
        # play a dev card
        # card must be in player's inventory
        if self.player_dev_cards[player_id].get(card_type, 0) <= 0:
            return False
        
        # can't play victory point cards (they're auto-revealed)
        if card_type == "victory_point":
            return False
        
        # can't play a card bought this turn
        cards_bought_this_turn = self.dev_cards_bought_this_turn.get(player_id, [])
        if card_type in cards_bought_this_turn:
            bought_count = cards_bought_this_turn.count(card_type)
            total_count = self.player_dev_cards[player_id].get(card_type, 0)
            
            # if all cards of this type were bought this turn, can't play any
            if bought_count >= total_count:
                if self.verbose >= 3:
                    print(f"  ❌ Player {self._player_display(player_id)} cannot play {card_type} - bought this turn")
                return False
        
        if self.dev_card_played_this_turn.get(player_id, False):
            if self.verbose >= 3:
                print(f"  ❌ Player {self._player_display(player_id)} already played a dev card this turn")
            return False
        
        self.player_dev_cards[player_id][card_type] -= 1
        self.dev_card_played_this_turn[player_id] = True
        
        if self.verbose >= 3:
            print(f"  📜 Player {self._player_display(player_id)} played a {card_type} card")
        
        if card_type == "knight":
            self.knights_played[player_id] += 1
            self.players[player_id].knights_played += 1
            self._update_largest_army()
            self.players[player_id].place_robber(self)
        
        if card_type == "road_building":
            roads_built = 0
            while roads_built < 2:
                valid_roads = self.get_valid_road_locations(player_id)
                if not valid_roads:
                    break
                edge_id = self.players[player_id].choose_road_location(self, valid_roads)
                if edge_id is None:
                    break
                self.build_road(player_id, edge_id)
                roads_built += 1
        
        if card_type == "monopoly":
            resource = self.players[player_id].choose_monopoly_resource(self)
            total_stolen = 0
            for pid in range(self.config.num_players):
                if pid != player_id:
                    amount = self.inventories[pid].get(resource, 0)
                    if amount > 0:
                        self.inventories[pid][resource] -= amount
                        self.inventories[player_id][resource] += amount
                        total_stolen += amount
            if self.verbose >= 3:
                print(f"    Player {self._player_display(player_id)} stole {total_stolen} {resource} cards")

        if card_type == "year_of_plenty":
            for _ in range(2):
                resource = self.players[player_id].choose_year_of_plenty_resource(self)
                self.inventories[player_id][resource] += 1
                if self.verbose >= 3:
                    print(f"    Player {self._player_display(player_id)} gained 1 {resource} card")
        
        return True
    
    def roll_dice(self, player_id):
        # Roll dice and distribute resources
        roll = sum(random.randint(1, self.config.die_sides) for _ in range(self.config.n_die))
        
        if self.verbose >= 2:
            print(f"\n🎲 Rolled: {bold_print(roll)}")
        
        if roll == 7:
            self._handle_seven(player_id)
        else:
            self._distribute_resources(roll)
        
        return roll
    
    def move_robber(self, player_id, hex_id):
        # move the robber to a new hex
        if hex_id not in self.get_valid_robber_locations():
            return False
        
        # remove robber from current location and place
        if self.robber_hex_id is not None:
            self.hexes[self.robber_hex_id].has_robber = False
        
        self.hexes[hex_id].has_robber = True
        self.robber_hex_id = hex_id
        
        if self.verbose >= 3:
            print(f"  🔴 Player {self._player_display(player_id)} moved robber to hex {hex_id}")
        
        # query player's victim choice
        victims = self._get_robber_victims(player_id, hex_id)
        
        if victims:
            victim = self.players[player_id].choose_steal_victim(self, victims)
            if victim is not None:
                self._steal_random_resource(player_id, victim)
        
        return True
    
    def discard_resources(self, player_id, resources):
        """Discard specified resources from player's hand"""
        # Validate that player has these resources
        for resource in resources:
            if self.inventories[player_id].get(resource, 0) <= 0:
                return False
        
        # Execute discard
        for resource in resources:
            self.inventories[player_id][resource] -= 1
            if self.verbose >= 3:
                print(f"    Player {self._player_display(player_id)} discarded 1 {resource}")
        
        return True
    
    # helpers
    
    def _pay_cost(self, player_id, building_type):
        # deduct cost from player's inventory
        cost = self.building_costs[building_type]
        
        if not self.can_afford(player_id, building_type):
            return False
        
        for resource, amount in cost.items():
            self.inventories[player_id][resource] -= amount
        
        return True
        
    def _handle_seven(self, current_player_id):
        # handle rolling a 7
        # discarding
        for player in self.players:
            total_cards = sum(self.inventories[player.id].values())
            if total_cards > 7:
                num_to_discard = total_cards // 2
                if self.verbose >= 3:
                    print(f"  🔴 Player {self._player_display(player.id)} must discard {num_to_discard} cards")
                player.handle_discard(self, num_to_discard)
        
        # robber placement
        self.players[current_player_id].place_robber(self)
    
    def _distribute_resources(self, roll):
        # distribute resources for a dice roll
        for hex_id, hex_obj in self.hexes.items():
            if hex_obj.roll_number == roll and not hex_obj.has_robber:
                for vertex_id in hex_obj.adjacent_vertices:
                    vertex = self.vertices[vertex_id]
                    if vertex.owner is not None:
                        resource_name = hex_obj.resource
                        amount = 2 if vertex.is_city else 1
                        self.inventories[vertex.owner][resource_name] += amount
                        if self.verbose >= 4:
                            print(f"  Player {self._player_display(vertex.owner)} gained {amount} {resource_name}")
    
    def _get_robber_victims(self, robber_player_id, hex_id):
        # get list of players who can be robbed
        victims = set()
        for vertex_id in self.hexes[hex_id].adjacent_vertices:
            vertex = self.vertices[vertex_id]
            if vertex.owner is not None and vertex.owner != robber_player_id:
                victims.add(vertex.owner)
        return list(victims)
    
    def _steal_random_resource(self, thief_id, victim_id):
        # steal a random resource from victim
        available = [r for r, count in self.inventories[victim_id].items() if count > 0]
        if available:
            stolen = random.choice(available)
            self.inventories[victim_id][stolen] -= 1
            self.inventories[thief_id][stolen] += 1
            if self.verbose >= 3:
                print(f"  🔴 Player {self._player_display(thief_id)} stole {stolen} from Player {self._player_display(victim_id)}")

    def _update_longest_road(self):
        """Update longest road achievement - OPTIMIZED"""
        # Only recalculate for players with dirty road networks
        if not self._road_network_dirty:
            return
        
        for pid in self._road_network_dirty:
            num_roads = len(self.players[pid].owned_roads)
            
            # Quick path: If player has < 5 roads, they can't have longest road
            if num_roads < 5:
                self._longest_road_cache[pid] = num_roads
                continue
            
            # Quick path: If player has fewer roads than current holder - 2, skip
            if self.longest_road_info[0] is not None:
                holder_roads = len(self.players[self.longest_road_info[0]].owned_roads)
                if num_roads < holder_roads - 2:
                    continue
            
            # Actually calculate (expensive)
            self._longest_road_cache[pid] = self._calculate_longest_road_fast(pid)
        
        self._road_network_dirty.clear()
        
        # Rest of update logic...
        max_length = max(self._longest_road_cache.values()) if self._longest_road_cache else 0
        
        if max_length >= 5:
            candidates = [p for p, length in self._longest_road_cache.items() if length == max_length]
            
            if len(candidates) == 1:
                new_holder = candidates[0]
                old_holder = self.longest_road_info[0]
                
                if new_holder != old_holder:
                    if old_holder is not None:
                        self.players[old_holder].victory_points -= 2
                    self.players[new_holder].victory_points += 2
                    self.longest_road_info = (new_holder, max_length)
                    if self.verbose >= 1:
                        print(f"🛣️ Player {self._player_display(new_holder)} now has the longest road ({max_length}) (+2 VP)")

    def _calculate_longest_road_fast(self, player_id):
        """Fast longest road calculation using endpoint optimization"""
        owned_edges = [eid for eid, e in self.edges.items() if e.owner == player_id]
        
        if not owned_edges:
            return 0
        
        # Build adjacency for road network
        road_graph = defaultdict(list)  # vertex -> [(edge_id, other_vertex)]
        for eid in owned_edges:
            edge = self.edges[eid]
            road_graph[edge.vertex1].append((eid, edge.vertex2))
            road_graph[edge.vertex2].append((eid, edge.vertex1))
        
        # Find endpoints (degree 1 vertices) - optimal DFS starting points
        endpoints = [v for v, edges in road_graph.items() if len(edges) == 1]
        
        # If no endpoints (all cycles), pick arbitrary starting points
        if not endpoints:
            endpoints = list(road_graph.keys())[:2]  # Just pick 2 vertices
        
        max_length = 0
        
        # DFS from each endpoint
        for start_vertex in endpoints:
            for start_edge, next_vertex in road_graph[start_vertex]:
                visited = set()
                length = self._dfs_road_length(start_edge, start_vertex, visited, player_id, road_graph)
                max_length = max(max_length, length)
        
        return max_length

    def _dfs_road_length(self, edge_id, came_from, visited, player_id, road_graph):
        """DFS for road length - optimized with adjacency dict"""
        visited.add(edge_id)
        edge = self.edges[edge_id]
        
        # Get next vertex
        next_vertex = edge.vertex2 if came_from == edge.vertex1 else edge.vertex1
        
        # Check if blocked by opponent settlement
        vertex_owner = self.vertices[next_vertex].owner
        if vertex_owner is not None and vertex_owner != player_id:
            visited.remove(edge_id)
            return 1
        
        # Try continuing from next_vertex
        max_continuation = 0
        for next_edge, other_vertex in road_graph[next_vertex]:
            if next_edge == edge_id or next_edge in visited:
                continue
            
            length = self._dfs_road_length(next_edge, next_vertex, visited, player_id, road_graph)
            max_continuation = max(max_continuation, length)
        
        visited.remove(edge_id)
        return 1 + max_continuation
    
    def _update_largest_army(self):
        # update largest army achievement
        max_knights = max(self.knights_played.values()) if self.knights_played else 0
        
        if max_knights >= 3:
            candidates = [p for p, count in self.knights_played.items() if count == max_knights]
            
            if len(candidates) == 1:
                new_holder = candidates[0]
                old_holder = self.largest_army_info[0]
                
                if new_holder != old_holder:
                    if old_holder is not None:
                        self.players[old_holder].victory_points -= 2
                    self.players[new_holder].victory_points += 2
                    self.largest_army_info = (new_holder, max_knights)
                    if self.verbose >= 1:
                        print(f"⚔️ Player {self._player_display(new_holder)} now has the largest army ({max_knights} knights) (+2 VP)")
    
    def get_buildings(self, player_id):
        settlements = [v for v in self.vertices.values() 
                        if v.owner == player_id and not v.is_city]
        cities = [v for v in self.vertices.values() 
                    if v.owner == player_id and v.is_city]
        return settlements, cities

    def calculate_total_score(self, player_id):
        # calculate total victory points for a player
        s, c = self.get_buildings(player_id)
        num_settlements, num_cities = len(s), len(c)
        score = num_settlements + 2 * num_cities + self.player_dev_cards[player_id]['victory_point']
        if self.longest_road_info[0] == player_id:
            score += 2
        if self.largest_army_info[0] == player_id:
            score += 2
        assert score == self.players[player_id].victory_points, f"player_id {player_id} score mismatch! {score} vs {self.players[player_id].victory_points}"
        return score

    def execute_trade(self, from_player_id, to_player_id, offer, request):
        # Validate that from_player has the offered resources
        for resource, amount in offer.items():
            if self.inventories[from_player_id].get(resource, 0) < amount:
                return False
        
        # Validate that to_player has the requested resources
        for resource, amount in request.items():
            if self.inventories[to_player_id].get(resource, 0) < amount:
                return False
        
        # Execute trade
        for resource, amount in offer.items():
            self.inventories[from_player_id][resource] -= amount
            self.inventories[to_player_id][resource] += amount
        
        for resource, amount in request.items():
            self.inventories[to_player_id][resource] -= amount
            self.inventories[from_player_id][resource] += amount
        
        if self.verbose >= 3:
            print(f"  🤝 Player {self._player_display(from_player_id)} traded with Player {self._player_display(to_player_id)}")
        return True
    
    def bank_trade(self, player_id, give_resource, receive_resource, amount=1):
        # trade with the bank
        best_ratio = 4
        
        # check player's port access
        for edge_id, port_info in self.ports.items():
            edge = self.edges[edge_id]
            # Player has access to port if they own a settlement/city on either vertex
            v1_owned = self.vertices[edge.vertex1].owner == player_id
            v2_owned = self.vertices[edge.vertex2].owner == player_id
            
            if v1_owned or v2_owned:
                if port_info['resource'] is None:  # Generic 3:1 port
                    best_ratio = min(best_ratio, 3)
                elif port_info['resource'] == give_resource:  # Specialized 2:1 port
                    best_ratio = 2
                    break  # Can't get better than 2:1
        
        # Calculate total cost
        total_cost = best_ratio * amount
        
        # Validate player has enough resources
        if self.inventories[player_id].get(give_resource, 0) < total_cost:
            return False
        
        # Execute trade
        self.inventories[player_id][give_resource] -= total_cost
        self.inventories[player_id][receive_resource] += amount
        
        if self.verbose >= 3:
            print(f"  🏦 Player {self._player_display(player_id)} traded {total_cost} {give_resource} for {amount} {receive_resource} ({best_ratio}:1)")
        
        return True
        
    # GAME LOOP
    
    def is_game_over(self):
        # Check for winner or turn limit
        if self.turn >= self.config.max_turns:
            return True
        
        if any(self.get_player_score(p) >= self.config.victory_points_to_win 
            for p in range(self.config.num_players)):
            return True
        
        return False

    def plot_state(self):
        # Increased figure height slightly to accommodate bigger font/layout
        fig = plt.figure(figsize=(22, 14.5)) 
        
        # Reduced top margin (top=0.98) to push content higher,
        # Increased bottom margin (bottom=0.01) to maximize space, 
        # and adjusted overall grid to use the space better.
        gs = fig.add_gridspec(1, 2, width_ratios=[7, 3.2], wspace=0.05,
                              left=0.02, right=0.98, top=0.98, bottom=0.01)
        
        ax_board = fig.add_subplot(gs[0, 0])
        
        # --- BOARD PLOTTING (UNCHANGED) ---
        background_img = mpimg.imread("pngs/background.png")
        x_size, y_size = 6.7, 6
        x_middle, y_middle = self.hexes[9].position
        
        x_offset = 0.5
        x_middle = x_middle + x_offset
        
        ax_board.imshow(background_img, extent=[x_middle - x_size/2, x_middle + x_size/2, 
                                                y_middle - y_size/2, y_middle + y_size/2],
            zorder=0, origin='upper')

        for hex_obj in self.hexes.values():
            x, y = hex_obj.position
            x = x + x_offset
            
            try:
                img_hex = mpimg.imread(f"pngs/{hex_obj.resource}.png")
                size = 1.05
                img_hex = img_hex[::-1, :, :]
                ax_board.imshow(img_hex, extent=[x - size/2, x + size/2, 
                                                 y - size/2, y + size/2], 
                        zorder=1, origin='upper')
            except:
                color = resource_colors.get(hex_obj.resource, 'lightgray')
                hex_vertices = []
                for i in range(6):
                    angle = np.pi / 3 * i + np.pi / 6
                    vx = x + 0.55 * np.cos(angle)
                    vy = y + 0.55 * np.sin(angle)
                    hex_vertices.append([vx, vy])
                hex_vertices.append(hex_vertices[0])
                hex_poly = plt.Polygon(hex_vertices, color=color, alpha=0.7, zorder=1)
                ax_board.add_patch(hex_poly)
            
            if hex_obj.roll_number != 7:
                ax_board.text(x, y, str(hex_obj.roll_number), 
                    color='white', fontsize=20, fontweight='bold',
                    ha='center', va='center', zorder=2,
                    bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))
            
            if hex_obj.has_robber:
                try:
                    img_robber = mpimg.imread(f"pngs/robber.png")
                    size = .5
                    ax_board.imshow(img_robber,
                                extent=[x - size/2, x + size/2,
                                        (y - size/2) + 0.3, (y + size/2) + 0.3],
                                zorder=4, origin='upper')
                except:
                    pass

        for edge in self.edges.values():
            v1 = self.vertices[edge.vertex1]
            v2 = self.vertices[edge.vertex2]
            x1, y1 = v1.position[0] + x_offset, v1.position[1]
            x2, y2 = v2.position[0] + x_offset, v2.position[1]
            
            if edge.owner is not None:
                color = self.players[edge.owner].color
                width = 4.5
                alpha = 0.9
            else:
                color = 'lightgray'
                width = 2
                alpha = 0.2
            
            ax_board.plot([x1, x2], [y1, y2], color=color, 
                linewidth=width, alpha=alpha, zorder=3)
        
        for edge_id, port_info in self.ports.items():
            edge = self.edges[edge_id]
            v1 = self.vertices[edge.vertex1]
            v2 = self.vertices[edge.vertex2]
            
            mid_x = (v1.position[0] + v2.position[0]) / 2 + x_offset
            mid_y = (v1.position[1] + v2.position[1]) / 2
            
            if port_info['resource'] is None:
                port_color = 'white'
                port_label = '3:1'
            else:
                port_color = resource_colors.get(port_info['resource'], 'white')
                port_label = '2:1'

            circle = plt.Circle((mid_x, mid_y), 0.17, color=port_color,
                            ec='black', linewidth=2, zorder=6, alpha=0.9)
            ax_board.add_patch(circle)
            ax_board.text(mid_x, mid_y, port_label, ha='center', va='center',
                fontsize=13, fontweight='bold', zorder=7)
        
        for vertex in self.vertices.values():
            x, y = vertex.position[0] + x_offset, vertex.position[1]
            
            if vertex.owner is not None:
                color = self.players[vertex.owner].color
                
                if vertex.is_city:
                    try:
                        img_building = mpimg.imread(f"building_pngs/city_{color}.png")
                        size = .38
                        ax_board.imshow(img_building, extent=[x - size/2, x + size/2, 
                                                            y - size/2, y + size/2], 
                                    zorder=5, origin='upper')
                    except:
                        ax_board.scatter(x, y, c=color, s=200, marker='s',
                                zorder=5, edgecolors='black', linewidth=2.5)
                else:
                    try:
                        img_building = mpimg.imread(f"building_pngs/settlement_{color}.png")
                        size = .38
                        ax_board.imshow(img_building, extent=[x - size/2, x + size/2, 
                                                            y - size/2, y + size/2], 
                                    zorder=5, origin='upper')
                    except:
                        ax_board.scatter(x, y, c=color, s=140, marker='o',
                                zorder=5, edgecolors='black', linewidth=2)
        
        ax_board.set_aspect('equal')
        ax_board.axis('off')
        ax_board.set_xlim(x_middle - x_size/2 - 0.2, x_middle + x_size/2 + 0.2)
        ax_board.set_ylim(y_middle - y_size/2 - 0.2, y_middle + y_size/2 + 0.2)

        # --- INFO PANEL RESTRUCTURED ---
        # Height Ratios adjusted: Stats (3.5), Ach/Status (1.5), Costs (2.5 - increased size)
        # hspace reduced to 0.05 to reduce vertical gaps
        gs_right = gs[0, 1].subgridspec(3, 2, height_ratios=[3.5, 1.5, 2.5], hspace=0.05, wspace=0.1)

        # 1. Player Stats (Spanning both columns of the top row)
        ax_players = fig.add_subplot(gs_right[0, :])
        ax_players.axis('off')
        
        player_stats_text = "PLAYER STATISTICS\n"
        player_stats_text += "═" * 50 + "\n"
        
        for player in self.players:
            vp = self.calculate_total_score(player.id)
            total_resources = sum(self.inventories[player.id].values())
            settlements = sum(1 for v in self.vertices.values() 
                            if v.owner == player.id and not v.is_city)
            cities = sum(1 for v in self.vertices.values() 
                        if v.owner == player.id and v.is_city)
            roads = len(player.owned_roads)
            
            player_class = type(player).__name__
            indicator = "▶" if player.id == self.current_player else " "
            
            player_stats_text += f"\n{indicator} Player {self._player_display(player.id)} ({player_class[:3]})\n"
            player_stats_text += f"   VP: {vp:<2}  •  Resources: {total_resources:<2}\n"
            
            inv = self.inventories[player.id]
            
            player_stats_text += f"   SET:{settlements:<2} CIT:{cities:<2} RD:{roads:<2} KNIGHTS:{player.knights_played:<2}\n"
            player_stats_text += f"   WHEAT:{inv['wheat']:<2} WOOD:{inv['wood']:<2} BRICK:{inv['brick']:<2} SHEEP:{inv['sheep']:<2} ORE:{inv['ore']:<2}\n"
            player_stats_text += "   " + "─" * 46 + "\n"
        
        ax_players.text(0.5, 0.5, player_stats_text,
                fontsize=18, family='monospace', fontweight='bold',
                ha='center', va='center',
                transform=ax_players.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF8E1', 
                        alpha=0.95, edgecolor='#8B4513', linewidth=2))
        
        # 2. Achievements (Middle Left)
        ax_achievements = fig.add_subplot(gs_right[1, 0])
        ax_achievements.axis('off')
        
        achievement_text = "ACHIEVEMENTS\n"
        achievement_text += "═" * 20 + "\n"
        
        lr_holder, lr_len = self.longest_road_info
        la_holder, la_amt = self.largest_army_info
        
        lr_str = f"P{self._player_display(lr_holder)} ({lr_len})" if lr_holder is not None else "--"
        la_str = f"P{self._player_display(la_holder)} ({la_amt})" if la_holder is not None else "--"

        achievement_text += f" Longest Road: {lr_str}\n"
        achievement_text += f" Largest Army: {la_str}"
        
        ax_achievements.text(0.5, 0.5, achievement_text,
                fontsize=18, family='monospace', fontweight='bold',
                ha='center', va='center',
                transform=ax_achievements.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF8E1', 
                        alpha=0.95, edgecolor='#8B4513', linewidth=2))
        
        # 3. Game Status (Middle Right)
        ax_state = fig.add_subplot(gs_right[1, 1])
        ax_state.axis('off')
        
        game_state_text = "GAME STATUS\n"
        game_state_text += "═" * 20 + "\n"
        
        total_dev_cards = len(self.dev_card_deck)
        game_state_text += f" Dev Cards: {total_dev_cards:<2}\n"
        
        if total_dev_cards > 0:
            c = Counter(self.dev_card_deck)
            game_state_text += f" Kt:{c.get('knight', 0):<2} VP:{c.get('victory_point', 0):<2} Mo:{c.get('monopoly', 0):<2}\n"
            game_state_text += f" RB:{c.get('road_building', 0):<2} YP:{c.get('year_of_plenty', 0):<2}\n"

        game_state_text += f" Goal: {self.config.victory_points_to_win} VP"
        
        ax_state.text(0.5, 0.5, game_state_text,
                fontsize=18, family='monospace', fontweight='bold',
                ha='center', va='center',
                transform=ax_state.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF8E1', 
                        alpha=0.95, edgecolor='#8B4513', linewidth=2))
        
        # 4. Costs Image (Spanning both columns of the bottom row, now taller)
        ax_costs = fig.add_subplot(gs_right[2, :])
        ax_costs.axis('off')
        
        try:
            costs_img = mpimg.imread("pngs/costs.png")
            # Using extent=[0, 1, 0, 1] fills the subplot space but aspect='equal' prevents squashing.
            # We set the subplot height ratio high enough (2.5) to give it room.
            ax_costs.imshow(costs_img, aspect='equal', extent=[0, 1, 0, 1])
        except FileNotFoundError:
            costs_text = "BUILDING COSTS\n" + "═" * 30 + "\n"
            costs_text += "Road: Wood, Brick\nSettlement: Wood, Brick, Wheat, Sheep\nCity: 2 Wheat, 3 Ore\nDev Card: Sheep, Wheat, Ore"
            ax_costs.text(0.5, 0.5, costs_text, fontsize=15, ha='center', va='center', family='monospace', fontweight='bold', color='#4A2511')
        except Exception:
            costs_text = "COSTS MISSING\n(Image Read Error)"
            ax_costs.text(0.5, 0.5, costs_text, fontsize=15, ha='center', va='center', color='red', fontweight='bold')

        # --- TITLE ---
        turn_info = f"Turn {self.turn} • Player {self._player_display(self.current_player)}'s Turn"
        # Since we set top=0.98 on the main grid, we need to adjust the title y-position down slightly (e.g., y=0.97)
        fig.suptitle(f"Settlers of Catan\n{turn_info}", 
                    fontsize=30, fontweight='bold', family='serif',
                    color='#4A2511', y=0.97)
        
        return fig
    
    def play_game(self):
        """
        Verbosity levels:
            0 - No output
            1 - Victory point events only (settlements, cities, longest road, largest army)
            2 - Turn-level actions (dice rolls, turn headers)
            3 - All actions (roads, dev cards, robber moves, trades, discards)
            4 - Everything including resource distributions
        """
        verbose = self.verbose

        if verbose >= 3:
            print("\n" + "="*60)
            print("STARTING CATAN GAME")
            print("="*60)
        
        self.setup_initial_placements()

        while not self.is_game_over():
            if verbose >= 2:
                print(f"\n{'='*60}")
                print(f"Turn {self.turn} - Player {self._player_display(self.current_player)}'s turn")
                print(f"{'='*60}")
            
            # clear dev card tracking at start of turn
            self.dev_cards_bought_this_turn[self.current_player] = []
            self.dev_card_played_this_turn[self.current_player] = False
            
            # request dice roll
            roll = self.roll_dice(self.current_player)

            # get current state
            if verbose >= 3:
                inventory = self.get_player_inventory(self.current_player)
                total_resources = sum(inventory.values())
                print(f"Player {self._player_display(self.current_player)} inventory: {inventory} (total: {total_resources})")

            self.players[self.current_player].turn(self)
            
            self.current_player = (self.current_player + 1) % self.config.num_players
            if self.current_player == 0:
                self.turn += 1
        
        # game over
        if verbose >= 1:
            print("\n" + "="*60)
            print("GAME OVER!")
            print("="*60)
        
        if verbose >= 1:
            print("\nFinal Scores:")
        scores = [(pid, self.get_player_score(pid)) 
                  for pid in range(self.config.num_players)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (pid, score) in enumerate(scores, 1):
            score_str = color_print(score) if verbose >= 2 else str(score)
            if verbose >= 1:
                print(f"{rank}. Player {self._player_display(pid)}: {score_str} VP")
            
            if verbose >= 2:
                settlements = sum(1 for v in self.vertices.values() 
                                if v.owner == pid and not v.is_city)
                cities = sum(1 for v in self.vertices.values() 
                            if v.owner == pid and v.is_city)
                
                print(f"   🏠 Settlements: {settlements}")
                if cities > 0:
                    print(f"   🏛️ Cities: {cities}")

                if self.longest_road_info[0] == pid:
                    print(f"   🛣️ Longest Road: +2")
                if self.largest_army_info[0] == pid:
                    print(f"   ⚔️ Largest Army: +2")
                if self.player_dev_cards[pid]['victory_point'] > 0:
                    print(f"   📜 Victory Point Cards: +{self.player_dev_cards[pid]['victory_point']}")
        
        winner = scores[0][0]
        if verbose >= 1:
            print(f"\n🏆 Player {self._player_display(winner)} wins!")
        
        return winner