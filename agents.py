import random
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
import math
import statistics
import time
from tqdm import tqdm

class Player:
    # Base player class, player subclasses implement choice methods
    def __init__(self, id: int, color: str):
        self.id = id
        self.color = color
        self.victory_points = 0
        self.development_cards = defaultdict(int)
        self.knights_played = 0
        self.owned_roads: Set[int] = set()
        self.owned_vertices: Set[int] = set()
        
        self.allow_trading = True
        self.allow_bank_trading = True

    def turn(self, game):
        #turn_start = time.time()
        MAX_ACTIONS_PER_TURN = 5
        attempts = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3
        failed_actions_this_turn = set()
        
        action_times = {}

        while attempts < MAX_ACTIONS_PER_TURN:
            attempts += 1
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
            
            possible_actions = self._get_possible_actions_strict(game)
            available_actions = [a for a in possible_actions if a not in failed_actions_this_turn]
            
            if not available_actions or available_actions == ['end_turn']:
                break
            
            action_start = time.time()
            action = self.choose_action(game, available_actions)
            action_choice_time = time.time() - action_start
            
            if action is None or action == 'end_turn':
                break
            
            action_exec_start = time.time()
            action_succeeded = False
            
            if action == 'build_city':
                action_succeeded = self._try_build_city(game)
            elif action == 'build_settlement':
                action_succeeded = self._try_build_settlement(game)
            elif action == 'build_road':
                action_succeeded = self._try_build_road(game)
            elif action == 'buy_dev_card':
                action_succeeded = self._try_buy_dev_card(game)
            elif action == 'play_dev_card':
                action_succeeded = self._try_play_dev_card(game)
            elif action == 'bank_trade':
                action_succeeded = self._try_bank_trade(game)
            
            action_exec_time = time.time() - action_exec_start
            
            # track timing for debugging bottlenecks
            if action not in action_times:
                action_times[action] = []
            action_times[action].append({
                'choice': action_choice_time,
                'exec': action_exec_time,
                'success': action_succeeded
            })
            
            if action_succeeded:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                failed_actions_this_turn.add(action)
        
    def _get_possible_actions(self, game):
        # self explanatory
        possible_actions = []
        
        if game.can_afford(self.id, "city") and game.get_valid_city_locations(self.id):
            possible_actions.append('build_city')
        
        if game.can_afford(self.id, "settlement") and game.get_valid_settlement_locations(self.id):
            possible_actions.append('build_settlement')
        
        if game.can_afford(self.id, "road") and game.get_valid_road_locations(self.id):
            possible_actions.append('build_road')
        
        if game.can_afford(self.id, "development_card"):
            possible_actions.append('buy_dev_card')
        
        playable_cards = self._get_playable_dev_cards(game)
        if playable_cards:
            possible_actions.append('play_dev_card')
        
        # Include trading options if enabled--let agent decide when to trade
        can_trade = game.config.allow_trading and self.allow_trading
        if can_trade:
            possible_actions.append('trade')
        
        can_bank_trade = game.config.allow_bank_trading and self.allow_bank_trading
        if can_bank_trade:
            if self._can_bank_trade(game):
                possible_actions.append('bank_trade')
        
        # ending turns is always allowed
        possible_actions.append('end_turn')
        
        return possible_actions
    
    def _get_possible_actions_strict(self, game):
        actions = []
        
        inv = game.get_player_inventory(self.id)
        
        can_afford_settlement = game.can_afford(self.id, 'settlement')
        valid_settlement_locs = game.get_valid_settlement_locations(self.id)
        if can_afford_settlement and valid_settlement_locs:
            actions.append('build_settlement')
        
        can_afford_city = game.can_afford(self.id, 'city')
        valid_city_locs = game.get_valid_city_locations(self.id)
        if can_afford_city and valid_city_locs:
            actions.append('build_city')
        
        can_afford_road = game.can_afford(self.id, 'road')
        valid_road_locs = game.get_valid_road_locations(self.id)
        if can_afford_road and valid_road_locs:
            actions.append('build_road')
        
        if game.can_afford(self.id, 'development_card') and len(game.dev_card_deck) > 0:
            actions.append('buy_dev_card')
        
        playable = self._get_playable_dev_cards(game)
        if playable:
            actions.append('play_dev_card')

        if game.config.allow_bank_trading and self.allow_bank_trading:
            ratios = game.get_player_trade_ratios(self.id)
            can_trade = any(inv.get(res, 0) >= ratio for res, ratio in ratios.items())
            if can_trade:
                actions.append('bank_trade')
        
        # remove trading because it should be a strategic choice, not something that's "always possible"
        # The agent can still choose to end turn if they want to trade but can't
        # This prevents infinite loops where trade always fails, this kept happening
        
        actions.append('end_turn')
        
        return actions

    def _can_bank_trade(self, game):
        """Check if player has enough resources to make any bank trade (operational)"""
        inventory = game.get_player_inventory(self.id)
        trade_ratios = game.get_player_trade_ratios(self.id)
        
        # Check each resource to see if we have enough for a trade
        for resource, count in inventory.items():
            # Get the best ratio for this resource (default 4:1)
            ratio = trade_ratios.get(resource, 4)
            if count >= ratio:
                return True
        
        return False
    
    def _try_build_city(self, game):
        """Attempt to build a city (operational)"""
        if not game.can_afford(self.id, "city"):
            return False
        
        valid_locations = game.get_valid_city_locations(self.id)
        if not valid_locations:
            return False
        
        location = self.choose_city_location(game, valid_locations)
        if location is None:
            return False
        
        return game.build_city(self.id, location)
    
    def _try_build_settlement(self, game):
        """Attempt to build a settlement (operational)"""
        if not game.can_afford(self.id, "settlement"):
            return False
        
        valid_locations = game.get_valid_settlement_locations(self.id)
        if not valid_locations:
            return False
        
        location = self.choose_settlement_location(game, valid_locations)
        if location is None:
            return False
        
        return game.build_settlement(self.id, location)
    
    def _try_build_road(self, game):
        """Attempt to build a road (operational)"""
        if not game.can_afford(self.id, "road"):
            return False
        
        valid_locations = game.get_valid_road_locations(self.id)
        if not valid_locations:
            return False
        
        location = self.choose_road_location(game, valid_locations)
        if location is None:
            return False
        
        return game.build_road(self.id, location)
    
    def _try_buy_dev_card(self, game):
        """Attempt to buy a development card (operational)"""
        if not game.can_afford(self.id, "development_card"):
            return False
        
        return game.buy_development_card(self.id)
    
    def _get_playable_dev_cards(self, game):
        """Get list of development cards that can be played this turn (operational)"""
        playable = []
        
        # cant play dev cards received from this turn
        if game.dev_card_played_this_turn.get(self.id, False):
            return playable
        
        dev_cards = game.get_player_dev_cards(self.id)
        cards_bought_this_turn = game.dev_cards_bought_this_turn.get(self.id, [])
        
        for card_type, count in dev_cards.items():
            if card_type == 'victory_point':
                continue
            
            if card_type in ['knight', 'monopoly', 'year_of_plenty', 'road_building']:
                bought_this_turn = cards_bought_this_turn.count(card_type)
                playable_count = count - bought_this_turn
                
                if playable_count > 0:
                    playable.append(card_type)
        
        return playable
    
    def _try_play_dev_card(self, game):
        # attempt to play a development card
        playable_cards = self._get_playable_dev_cards(game)
        
        if not playable_cards:
            return False
        
        card_to_play = self.choose_dev_card_to_play(game, playable_cards)
        
        if card_to_play:
            success = game.play_development_card(self.id, card_to_play)
            return success
        
        return False
    
    def _try_bank_trade(self, game):
        # attempt to trade w/ bank
        if not self._can_bank_trade(game):
            return False
        
        # Get available resources and what we need
        inventory = game.get_player_inventory(self.id)
        trade_ratios = game.get_player_trade_ratios(self.id)
        
        # Build list of possible trades
        possible_trades = []
        for give_resource, give_count in inventory.items():
            ratio = trade_ratios.get(give_resource, 4)
            if give_count >= ratio:
                # Can trade this resource
                for receive_resource in ['wheat', 'wood', 'brick', 'sheep', 'ore']:
                    if receive_resource != give_resource:
                        possible_trades.append({
                            'give': give_resource,
                            'receive': receive_resource,
                            'ratio': ratio,
                            'max_amount': give_count // ratio
                        })
        
        if not possible_trades:
            return False
        
        # Let subclass choose which trade to make
        chosen_trade = self.choose_bank_trade(game, possible_trades)
        
        if chosen_trade is None:
            return False
        
        return game.bank_trade(
            self.id,
            chosen_trade['give'],
            chosen_trade['receive'],
            chosen_trade.get('amount', 1)
        )
    
    def _try_trade(self, game):
        """Attempt to initiate a trade with other players (operational)"""
        needed_resources = self.determine_needed_resources(game)
        
        if not needed_resources:
            return False
        
        available_resources = self.determine_tradeable_resources(game)
        
        if not available_resources:
            return False
        
        strategy = self.choose_trade_strategy(game)
        
        if strategy == 'targeted':
            target = self.choose_trade_target(game)
            trade_offer = self._create_trade_offer(needed_resources, available_resources)
            if trade_offer and target is not None:
                return self._pose_trade_to_player(game, trade_offer, target)
        
        elif strategy == 'broadcast':
            trade_offer = self._create_trade_offer(needed_resources, available_resources)
            if trade_offer:
                return self._pose_trade_to_all(game, trade_offer)
        
        elif strategy == 'multiple':
            trade_offers = self._create_multiple_trade_offers(needed_resources, available_resources)
            if trade_offers:
                return self._pose_multiple_trades(game, trade_offers)
        
        return False
    
    def _pose_trade_to_player(self, game, trade_offer, target_player_id):
        #Pose a trade offer to a specific player
        if target_player_id == self.id:
            return False
        
        my_id = game._player_display(self.id)
        target_id = game._player_display(target_player_id)
        
        if game.verbose >= 2:
            print(f"Player {my_id} proposes to Player {target_id}: {trade_offer}")
        
        target_player = game.players[target_player_id]
        accepts = target_player.accept_trade_offer(game, trade_offer['offering'], trade_offer['requesting'])
        
        if accepts:
            if game.execute_trade(self.id, target_player_id, trade_offer['offering'], trade_offer['requesting']):
                if game.verbose >= 2:
                    print(f"Player {target_id} accepted the trade!")
                return True
            else:
                if game.verbose >= 2:
                    print(f"Player {target_id} considered the trade but it didn't work out")
        else:
            if game.verbose >= 2:
                print(f"Player {target_id} declined the trade")
            pass
        
        return False
    
    def _pose_trade_to_all(self, game, trade_offer):
        """Pose a trade offer to all other players (operational)"""
        my_id = game._player_display(self.id)
        
        if game.verbose >= 2:
            print(f"Player {my_id} proposes to everyone: {trade_offer}")
        
        other_players = [p for p in range(game.config.num_players) if p != self.id]
        random.shuffle(other_players)
        
        for player_id in other_players:
            player = game.players[player_id]
            accepts = player.accept_trade_offer(game, trade_offer['offering'], trade_offer['requesting'])
            
            if accepts:
                if game.execute_trade(self.id, player_id, trade_offer['offering'], trade_offer['requesting']):
                    if game.verbose >= 2:
                        print(f"Player {game._player_display(player_id)} accepted the trade!")
                    return True
                else:
                    if game.verbose >= 2:
                        print(f"Player {game._player_display(player_id)} considered the trade but it didn't work out")
        if game.verbose >= 2:
            print(f"No one accepted Player {my_id}'s trade")
        return False
    
    def _pose_multiple_trades(self, game, trade_offers):
        """Try multiple trade offers in sequence (operational)"""
        my_id = game._player_display(self.id)
        
        for i, trade_offer in enumerate(trade_offers):
            if game.verbose >= 2:
                print(f"Player {my_id} trying trade option {i+1}/{len(trade_offers)}")
            if self._pose_trade_to_all(game, trade_offer):
                return True
        return False
    
    def _create_trade_offer(self, needed_resources, available_resources):
        """Create a single trade offer (operational)"""
        if not needed_resources or not available_resources:
            return None
        
        most_needed = needed_resources.most_common(1)[0][0]
        need_amount = 1
        
        most_available = available_resources.most_common(1)[0][0]
        offer_amount = min(2, available_resources[most_available])
        
        return {
            'offering': {most_available: offer_amount},
            'requesting': {most_needed: need_amount}
        }
    
    def _create_multiple_trade_offers(self, needed_resources, available_resources):
        """Create multiple trade offer alternatives (operational)"""
        offers = []
        
        needed_list = needed_resources.most_common(3)
        available_list = available_resources.most_common(3)
        
        for needed_resource, _ in needed_list:
            for available_resource, avail_count in available_list:
                if needed_resource != available_resource:
                    offer_amount = min(2, avail_count)
                    offers.append({
                        'offering': {available_resource: offer_amount},
                        'requesting': {needed_resource: 1}
                    })
        
        return offers[:3]
    
    def determine_needed_resources(self, game):
        raise NotImplementedError("Subclass must implement determine_needed_resources")
    
    def determine_tradeable_resources(self, game):
        raise NotImplementedError("Subclass must implement determine_tradeable_resources")
    
    def setup_initial_placements(self, game, round_num):
        """Make decisions for initial settlement and road placements (operational)"""
        available_vertices = game.get_valid_settlement_locations(self.id, initial_placement=True)
        
        if not available_vertices:
            print(f"WARNING: Player {game._player_display(self.id)} has no valid settlement locations!")
            return
        
        settlement_vertex = self.choose_initial_settlement(game, available_vertices, round_num)
        
        if game.build_settlement(self.id, settlement_vertex, initial_placement=True, 
                                second_placement=(round_num == 2)):
            available_edges = game.get_valid_road_locations(self.id, initial_placement=True, 
                                                           specific_vertex=settlement_vertex)
            
            if available_edges:
                edge_id = self.choose_initial_road(game, available_edges, settlement_vertex, round_num)
                
                if game.build_road(self.id, edge_id, initial_placement=True):
                    pass
    
    def handle_discard(self, game, num_to_discard):
        """Choose which resources to discard when rolled 7 (operational)"""
        inventory = game.get_player_inventory(self.id)
        all_resources = []
        for resource, count in inventory.items():
            all_resources.extend([resource] * count)
        resources_to_discard = self.choose_resources_to_discard(game, all_resources, num_to_discard)
        return game.discard_resources(self.id, resources_to_discard)
    
    def place_robber(self, game):
        """Choose where to place the robber (operational)"""
        valid_hexes = game.get_valid_robber_locations()
        
        if not valid_hexes:
            return False
        
        chosen_hex = self.choose_robber_placement(game, valid_hexes)
        
        if chosen_hex is not None:
            return game.move_robber(self.id, chosen_hex)
        
        return False
    
    def choose_steal_victim(self, game, possible_victims):
        if possible_victims:
            return random.choice(possible_victims)
        return None
    
    def choose_action(self, game, possible_actions):
        raise NotImplementedError("Subclass must implement choose_action")
    
    def choose_bank_trade(self, game, possible_trades):
        raise NotImplementedError("Subclass must implement choose_bank_trade")
    
    def choose_city_location(self, game, valid_locations):
        raise NotImplementedError("Subclass must implement choose_city_location")
    
    def choose_settlement_location(self, game, valid_locations):
        raise NotImplementedError("Subclass must implement choose_settlement_location")
    
    def choose_road_location(self, game, valid_locations):
        raise NotImplementedError("Subclass must implement choose_road_location")
    
    def choose_dev_card_to_play(self, game, playable_cards):
        raise NotImplementedError("Subclass must implement choose_dev_card_to_play")
    
    def choose_trade_strategy(self, game):
        raise NotImplementedError("Subclass must implement choose_trade_strategy")
    
    def choose_trade_target(self, game):
        raise NotImplementedError("Subclass must implement choose_trade_target")
    
    def accept_trade_offer(self, game, offering, requesting):
        raise NotImplementedError("Subclass must implement accept_trade_offer")
    
    def choose_initial_settlement(self, game, valid_locations, round_num):
        raise NotImplementedError("Subclass must implement choose_initial_settlement")
    
    def choose_initial_road(self, game, valid_edges, settlement_vertex, round_num):
        raise NotImplementedError("Subclass must implement choose_initial_road")
    
    def choose_resources_to_discard(self, game, all_resources, num_to_discard):
        raise NotImplementedError("Subclass must implement choose_resources_to_discard")
    
    def choose_robber_placement(self, game, valid_hexes):
        raise NotImplementedError("Subclass must implement choose_robber_placement")
    
    def choose_monopoly_resource(self, game):
        raise NotImplementedError("Subclass must implement choose_monopoly_resource")
    
    def choose_year_of_plenty_resource(self, game):
        raise NotImplementedError("Subclass must implement choose_year_of_plenty_resource")
    
class RandomAgent(Player):
    """Simple random agent that only implements choice methods"""
    
    def __init__(self, id: int, color: str):
        super().__init__(id, color)

        self.allow_trading = True 
        self.allow_bank_trading = True

    def choose_action(self, game, possible_actions):
        """Randomly choose an action"""
        return random.choice(possible_actions) if possible_actions else None
    
    def choose_city_location(self, game, valid_locations):
        return random.choice(valid_locations)
    
    def choose_settlement_location(self, game, valid_locations):
        return random.choice(valid_locations)
    
    def choose_road_location(self, game, valid_locations):
        return random.choice(valid_locations)
    
    def choose_dev_card_to_play(self, game, playable_cards):
        return random.choice(playable_cards) if random.random() < 0.5 else None
    
    def choose_trade_strategy(self, game):
        return random.choice(['targeted', 'broadcast', 'multiple'])
    
    def choose_trade_target(self, game):
        candidates = [i for i in range(game.config.num_players) if i != self.id]
        return random.choice(candidates) if candidates else None
    
    def accept_trade_offer(self, game, offering, requesting):
        inventory = game.get_player_inventory(self.id)
        can_afford = all(inventory.get(resource, 0) >= count 
                        for resource, count in requesting.items())
        return can_afford and random.choice([True, False])
    
    def choose_bank_trade(self, game, possible_trades):
        return random.choice(possible_trades)

    def choose_initial_settlement(self, game, valid_locations, round_num):
        return random.choice(valid_locations)
    
    def choose_initial_road(self, game, valid_edges, settlement_vertex, round_num):
        return random.choice(valid_edges)
    
    def choose_resources_to_discard(self, game, all_resources, num_to_discard):
        random.shuffle(all_resources)
        return all_resources[:num_to_discard]
    
    def choose_robber_placement(self, game, valid_hexes):
        return random.choice(valid_hexes)
    
    def choose_monopoly_resource(self, game):
        return random.choice(['ore', 'wheat', 'brick', 'wood', 'sheep'])
    
    def choose_year_of_plenty_resource(self, game):
        return random.choice(['ore', 'wheat', 'brick', 'wood', 'sheep'])
    
    def choose_steal_victim(self, game, possible_victims):
        return random.choice(possible_victims) if possible_victims else None
    
class GreedyAgent(Player):
    """Intelligent greedy agent that makes strategic decisions"""

    def __init__(self, id: int, color: str):
        super().__init__(id, color)
        self.DIVERSITY_WEIGHT = 0.15
        self.PORT_BONUS = 0.3
        self.EXPANSION_WEIGHT = 0.2
        self.TRADE_FREQUENCY = 0.6

        self.allow_trading = True 
        self.allow_bank_trading = True
    
    def choose_action(self, game, possible_actions):
        # choose action queue based on priority, kinda arbitrary

        # dev cards first
        if 'play_dev_card' in possible_actions:
            playable_cards = self._get_playable_dev_cards(game)
            if 'knight' in playable_cards and self.should_play_knight(game):
                return 'play_dev_card'
            if 'monopoly' in playable_cards and self.should_play_monopoly(game):
                return 'play_dev_card'
            if 'year_of_plenty' in playable_cards and self.should_play_year_of_plenty(game):
                return 'play_dev_card'
            if 'road_building' in playable_cards and self.should_play_road_building(game):
                return 'play_dev_card'
        
        # building next
        priority_order = ['build_city', 'build_settlement', 'build_road', 'buy_dev_card']
        for action in priority_order:
            if action in possible_actions:
                return action
            
        # bank trade
        if 'bank_trade' in possible_actions:
            return 'bank_trade'
        
        # regular trade
        if 'trade' in possible_actions and random.random() < self.TRADE_FREQUENCY:
            return 'trade'
        
        return 'end_turn'

    def choose_dev_card_to_play(self, game, playable_cards):
        if 'knight' in playable_cards and self.should_play_knight(game):
            return 'knight'
        if 'monopoly' in playable_cards and self.should_play_monopoly(game):
            return 'monopoly'
        if 'year_of_plenty' in playable_cards and self.should_play_year_of_plenty(game):
            return 'year_of_plenty'
        if 'road_building' in playable_cards and self.should_play_road_building(game):
            return 'road_building'
        return None

    def should_play_knight(self, game):
        our_knights = self.knights_played
        try:
            largest_army_holder = game.get_largest_army_holder()
            if largest_army_holder is not None:
                their_knights = game.players[largest_army_holder].knights_played
                if our_knights >= their_knights - 1:
                    return True
        except:
            pass
        
        our_score = game.get_player_score(self.id)
        all_scores = [game.get_player_score(i) for i in range(game.config.num_players)]
        leader_score = max(all_scores)
        
        if our_score < leader_score - 1:
            return random.random() < 0.6
        
        return random.random() < 0.3
    
    def should_play_monopoly(self, game):
        inventory = game.get_player_inventory(self.id)
        
        if not game.can_afford(self.id, "city"):
            ore_needed = max(0, 3 - inventory.get("ore", 0))
            wheat_needed = max(0, 2 - inventory.get("wheat", 0))
            
            if ore_needed > 0 and ore_needed <= 2 and wheat_needed == 0:
                return True
            if wheat_needed > 0 and wheat_needed <= 2 and ore_needed == 0:
                return True
        
        if not game.can_afford(self.id, "settlement"):
            needed = self.determine_needed_resources(game)
            most_needed = needed.most_common(1)
            if most_needed and most_needed[0][1] >= 1:
                return random.random() < 0.4
        
        return False

    def should_play_year_of_plenty(self, game):
        needed = self.determine_needed_resources(game)
        
        if self._can_almost_afford(game, "city"):
            return True
        if self._can_almost_afford(game, "settlement"):
            return True
        if needed:
            return random.random() < 0.3
        
        return False

    def should_play_road_building(self, game):
        try:
            longest_road_holder = game.get_longest_road_holder()
            our_roads = len(self.owned_roads)
            
            if longest_road_holder is not None:
                their_roads = len(game.players[longest_road_holder].owned_roads)
                if our_roads >= their_roads - 2:
                    return True
        except:
            pass
        
        potential_settlements = game.get_valid_settlement_locations(self.id)
        if potential_settlements:
            return random.random() < 0.4
        
        return random.random() < 0.2

    def choose_monopoly_resource(self, game):
        needed = self.determine_needed_resources(game)
        if needed:
            most_needed = needed.most_common(1)[0][0]
            return most_needed
        return random.choice(["ore", "wheat"])

    def choose_year_of_plenty_resource(self, game):
        needed = self.determine_needed_resources(game)
        if needed:
            most_needed = needed.most_common(2)
            resource_options = [res for res, _ in most_needed]
            return random.choice(resource_options) if resource_options else "wheat"
        return "wheat"

    def choose_city_location(self, game, valid_locations):
        best_location = None
        best_score = -1
        
        for location in valid_locations:
            score = self._calculate_vertex_resource_score(game, location)
            if score > best_score:
                best_score = score
                best_location = location
        
        return best_location if best_location else random.choice(valid_locations)
    
    def choose_settlement_location(self, game, valid_locations):
        valid_locations_scores = {}
        current_resources = self._get_resource_distribution(game)

        for location in valid_locations:
            score = 0
            adjacent_hexes = game.vertices[location].adjacent_hexes
            resource_types = set()
            
            for hex_id in adjacent_hexes:
                hex_info = game.get_hex_info(hex_id)
                if hex_info['resource'] != "desert":
                    prob = self._get_probability(game, hex_info['roll_number'])
                    
                    scarcity_bonus = 1.0
                    if hex_info['resource'] in current_resources:
                        total_resources = sum(current_resources.values())
                        if total_resources > 0:
                            resource_ratio = current_resources[hex_info['resource']] / total_resources
                            scarcity_bonus = 1.0 + (1.0 - resource_ratio) * 0.3
                    else:
                        scarcity_bonus = 1.3
                    
                    score += prob * scarcity_bonus
                    resource_types.add(hex_info['resource'])
            
            score += len(resource_types) * self.DIVERSITY_WEIGHT
            port_score = self._calculate_port_score(game, location)
            score += port_score * self.PORT_BONUS
            
            valid_locations_scores[location] = score
        
        return max(valid_locations_scores, key=valid_locations_scores.get)

    def choose_road_location(self, game, valid_locations):
        best_edge = None
        best_score = -1
        
        potential_settlements = game.get_valid_settlement_locations(self.id)
        settlement_scores = {}
        
        if potential_settlements:
            for settlement_loc in potential_settlements:
                settlement_scores[settlement_loc] = self._calculate_vertex_resource_score(game, settlement_loc)
        
        for edge in valid_locations:
            score = 0
            
            for vertex in {game.edges[edge].vertex1, game.edges[edge].vertex2}:
                if vertex in settlement_scores:
                    score += settlement_scores[vertex]
                
                port_score = self._calculate_port_score(game, vertex)
                score += port_score * self.EXPANSION_WEIGHT
            
            adjacent_edges = game.edges[edge].adjacent_edges
            our_adjacent_roads = sum(1 for adj_edge in adjacent_edges 
                                    if game.edges[adj_edge].owner == self.id)
            score += our_adjacent_roads * 0.1
            
            if score > best_score:
                best_score = score
                best_edge = edge
        
        return best_edge if best_edge else random.choice(valid_locations)

    def choose_trade_strategy(self, game):
        our_score = game.get_player_score(self.id)
        all_scores = [game.get_player_score(i) for i in range(game.config.num_players)]
        leader_score = max(all_scores)
        
        if our_score < leader_score - 2:
            return 'targeted'
        if leader_score - min(all_scores) <= 3:
            return 'broadcast'
        return 'multiple'

    def choose_trade_target(self, game):
        our_score = game.get_player_score(self.id)
        candidates = []
        
        for player_id in range(game.config.num_players):
            if player_id == self.id:
                continue
            score = game.get_player_score(player_id)
            if abs(score - our_score) <= 2:
                candidates.append(player_id)
        
        return random.choice(candidates) if candidates else None

    def accept_trade_offer(self, game, offering, requesting):
        inventory = game.get_player_inventory(self.id)
        
        can_afford = all(inventory.get(resource, 0) >= count 
                         for resource, count in requesting.items())
        
        if not can_afford:
            if random.random() < 0.3:
                pass
            return False
        
        trade_value = self._evaluate_trade_value(game, offering, requesting)
        
        try:
            proposer_id = getattr(game, 'current_trade_proposer', None)
            if proposer_id is not None:
                proposer_score = game.get_player_score(proposer_id)
            else:
                proposer_score = self._estimate_proposer_threat(game)
        except:
            proposer_score = self._estimate_proposer_threat(game)
        
        our_score = game.get_player_score(self.id)
        
        if proposer_score > our_score + 2:
            trade_value -= 0.3
        if proposer_score >= game.config.victory_points_to_win - 2:
            trade_value -= 0.5
        
        our_need_offered = self._calculate_resource_need(game, offering)
        our_need_requested = self._calculate_resource_need(game, requesting)
        
        if our_need_offered > our_need_requested * 1.5:
            trade_value += 0.3
        elif our_need_requested > our_need_offered * 1.5:
            trade_value -= 0.2
        
        trade_value += random.uniform(-0.15, 0.15)
        
        if trade_value > 0.0:
            if random.random() < 0.2:
                return False
            return True
        
        if trade_value > -0.3:
            if random.random() < 0.1:
                return True
        
        return False
    
    def choose_initial_settlement(self, game, valid_locations, round_num):
        vertex_scores = self._calculate_vertex_scores(game)
        return max(valid_locations, key=lambda v: vertex_scores.get(v, 0))

    def choose_initial_road(self, game, valid_edges, settlement_vertex, round_num):
        best_edge = None
        best_score = -1
        
        for edge in valid_edges:
            score = 0
            
            target_vertex = None
            for vertex in {game.edges[edge].vertex1, game.edges[edge].vertex2}:
                if vertex != settlement_vertex:
                    target_vertex = vertex
                    break
            
            if target_vertex:
                score = self._calculate_vertex_resource_score(game, target_vertex)
                port_score = self._calculate_port_score(game, target_vertex)
                score += port_score * self.PORT_BONUS
            
            if score > best_score:
                best_score = score
                best_edge = edge
        
        return best_edge if best_edge else random.choice(valid_edges)

    def choose_resources_to_discard(self, game, all_resources, num_to_discard):
        inventory = game.get_player_inventory(self.id)
        needed_resources = self.determine_needed_resources(game)
        
        resource_priority = []
        for resource, count in inventory.items():
            priority = needed_resources.get(resource, 0)
            for _ in range(count):
                resource_priority.append((resource, priority))
        
        resource_priority.sort(key=lambda x: x[1])
        return [resource for resource, _ in resource_priority[:num_to_discard]]

    def choose_robber_placement(self, game, valid_hexes):
        our_score = game.get_player_score(self.id)
        all_scores = [(i, game.get_player_score(i)) for i in range(game.config.num_players)]
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        leader_id = all_scores[0][0]
        our_position = next(i for i, (pid, _) in enumerate(all_scores) if pid == self.id)
        
        if our_position == 0:
            target_player = all_scores[1][0] if len(all_scores) > 1 else leader_id
        else:
            target_player = leader_id
        
        hex_scores = {}
        for hex_id in valid_hexes:
            score = self._calculate_robber_hex_score(game, hex_id, target_player)
            hex_scores[hex_id] = score
        
        return max(hex_scores, key=hex_scores.get)

    def choose_steal_victim(self, game, possible_victims):
        if not possible_victims:
            return None
        
        best_victim = None
        best_score = -1
        needed_resources = self.determine_needed_resources(game)
        
        for victim_id in possible_victims:
            score = 0
            
            victim_inventory = game.get_player_inventory(victim_id)
            total_resources = sum(victim_inventory.values())
            score += total_resources
            
            victim_score = game.get_player_score(victim_id)
            our_score = game.get_player_score(self.id)
            if victim_score > our_score:
                score += (victim_score - our_score) * 2
            
            for resource in needed_resources:
                if victim_inventory.get(resource, 0) > 0:
                    score += needed_resources[resource] * 0.5
            
            if score > best_score:
                best_score = score
                best_victim = victim_id
        
        return best_victim if best_victim is not None else random.choice(possible_victims)

    def choose_bank_trade(self, game, possible_trades):
        """Choose bank trade based on what we need for buildings"""
        inventory = game.get_player_inventory(self.id)
        
        settlement_needs = {'wood': 1, 'brick': 1, 'sheep': 1, 'wheat': 1}

        needed = []
        for resource, amount_needed in settlement_needs.items():
            if inventory.get(resource, 0) < amount_needed:
                needed.append(resource)
        
        if needed:
            good_trades = [t for t in possible_trades if t['receive'] in needed]
            
            if good_trades:
                good_trades.sort(key=lambda t: t['ratio'])
                best_trade = good_trades[0]
                best_trade['amount'] = 1
                return best_trade
            
        trade = random.choice(possible_trades)
        trade['amount'] = 1
        return trade

    def determine_needed_resources(self, game):
        needed = Counter()
        inventory = game.get_player_inventory(self.id)
        
        building_costs = {
            'city': {"ore": 3, "wheat": 2},
            'settlement': {"brick": 1, "wood": 1, "sheep": 1, "wheat": 1},
            'road': {"brick": 1, "wood": 1},
            'development_card': {"ore": 1, "sheep": 1, "wheat": 1}
        }
        
        priorities = ['city', 'settlement', 'road', 'development_card']
        
        for building_type in priorities:
            costs = building_costs.get(building_type, {})
            
            for resource, required in costs.items():
                current = inventory.get(resource, 0)
                if current < required:
                    shortage = required - current
                    weight = shortage * (2.0 if self._can_almost_afford(game, building_type) else 0.5)
                    needed[resource] += weight
        
        return needed

    def determine_tradeable_resources(self, game):
        inventory = game.get_player_inventory(self.id)
        tradeable = Counter()
        needed = self.determine_needed_resources(game)
        
        for resource, count in inventory.items():
            reserve = max(1, int(needed.get(resource, 0)))
            if count > reserve:
                tradeable[resource] = count - reserve
        
        return tradeable

    def _evaluate_trade_value(self, game, offering, requesting):
        value = 0.0
        our_needs = self.determine_needed_resources(game)
        
        for resource, count in offering.items():
            need_weight = our_needs.get(resource, 0.1)
            value += count * need_weight
        
        for resource, count in requesting.items():
            need_weight = our_needs.get(resource, 0.1)
            value -= count * (need_weight + 0.5)
        
        return value

    def _calculate_resource_need(self, game, resources):
        needed = self.determine_needed_resources(game)
        total_need = 0.0
        
        for resource, count in resources.items():
            need_weight = needed.get(resource, 0.1)
            total_need += count * need_weight
        
        return total_need

    def _estimate_proposer_threat(self, game):
        all_scores = [game.get_player_score(i) for i in range(game.config.num_players)]
        sorted_scores = sorted(all_scores, reverse=True)
        return sorted_scores[len(sorted_scores)//2]

    def _calculate_vertex_scores(self, game):
        vertex_scores = {}
        vertices = game.get_all_vertices()
        
        for vertex_id, vertex_data in vertices.items():
            score = 0
            resource_types = set()
            
            for hex_id in vertex_data['adjacent_hexes']:
                hex_data = game.get_hex_info(hex_id)
                if hex_data['resource'] != "desert":
                    prob = self._get_probability(game, hex_data['roll_number'])
                    score += prob
                    resource_types.add(hex_data['resource'])
            
            score += len(resource_types) * self.DIVERSITY_WEIGHT
            port_score = self._calculate_port_score(game, vertex_id)
            score += port_score * self.PORT_BONUS
            
            vertex_scores[vertex_id] = score
        
        return vertex_scores

    def _calculate_vertex_resource_score(self, game, vertex):
        score = 0
        adjacent_hexes = game.vertices[vertex].adjacent_hexes
        
        for hex_id in adjacent_hexes:
            hex_info = game.get_hex_info(hex_id)
            if hex_info['resource'] != "desert":
                prob = self._get_probability(game, hex_info['roll_number'])
                score += prob
        
        return score

    def _calculate_port_score(self, game, vertex):
        try:
            port_info = game.get_vertex_port_info(vertex)
            if port_info:
                if port_info['ratio'] == 2:
                    return 0.5
                elif port_info['ratio'] == 3:
                    return 0.3
        except:
            pass
        return 0.0

    def _get_resource_distribution(self, game):
        resource_dist = Counter()
        
        try:
            settlements = game.get_player_settlements(self.id)
            cities = game.get_player_cities(self.id)
            
            for settlement in settlements:
                adjacent_hexes = game.vertices[settlement].adjacent_hexes
                for hex_id in adjacent_hexes:
                    hex_info = game.get_hex_info(hex_id)
                    if hex_info['resource'] != "desert":
                        prob = self._get_probability(game, hex_info['roll_number'])
                        resource_dist[hex_info['resource']] += prob
            
            for city in cities:
                adjacent_hexes = game.vertices[city].adjacent_hexes
                for hex_id in adjacent_hexes:
                    hex_info = game.get_hex_info(hex_id)
                    if hex_info['resource'] != "desert":
                        prob = self._get_probability(game, hex_info['roll_number'])
                        resource_dist[hex_info['resource']] += prob * 2
        except:
            pass
        
        return resource_dist

    def _calculate_robber_hex_score(self, game, hex_id, target_player):
        score = 0
        
        try:
            hex_info = game.get_hex_info(hex_id)
            
            if hex_info['resource'] != "desert":
                prob = self._get_probability(game, hex_info['roll_number'])
                score += prob
            
            adjacent_vertices = game.get_hex_adjacent_vertices(hex_id)
            target_impact = 0
            our_impact = 0

            for vertex in adjacent_vertices:
                owner = game.get_vertex_owner(vertex)
                if owner == target_player:
                    building_type = game.get_vertex_building_type(vertex)
                    if building_type == 'city':
                        target_impact += 2
                    elif building_type == 'settlement':
                        target_impact += 1
                elif owner == self.id:
                    building_type = game.get_vertex_building_type(vertex)
                    if building_type == 'city':
                        our_impact += 2
                    elif building_type == 'settlement':
                        our_impact += 1
            
            score = score * target_impact - (score * our_impact * 0.5)
        except:
            pass
        
        return score

    def _can_almost_afford(self, game, build_type):
        try:
            inventory = game.get_player_inventory(self.id)
            costs = game.get_build_cost(build_type)
            
            resources_needed = 0
            for resource, cost in costs.items():
                current = inventory.get(resource, 0)
                if current < cost:
                    resources_needed += (cost - current)
            
            return resources_needed <= 2
        except:
            return False

    def _get_probability(self, game, roll_number):
        #try:
            #return probability_sum_equals(roll_number, game.config.n_die, game.config.die_sides)
        #except:
        probabilities = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 6/36,
                        8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}
        return probabilities.get(roll_number, 0)