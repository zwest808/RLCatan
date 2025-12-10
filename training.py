from helpers import *
from game import *
from agents import *

import os
import sys
import torch
import copy

import pickle
import gzip
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import random
from collections import deque
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from dataclasses import dataclass

@dataclass
class RewardConfig:
    vp_gain: float = 50.0
    vp_loss: float = 0.0

    settlement_built: float = 0.0
    city_built: float = 0.0
    road_built: float = 0.0

    win_reward: float = 1000.0
    loss_reward: float = 0.0

    longest_road_gained: float = 0.0
    longest_road_lost: float = 0.0
    largest_army_gained: float = 0.0
    largest_army_lost: float = 0.0
    dev_card_bought: float = 0.0
    knight_played: float = 0.0
    vp_card_revealed: float = 0.0
    resource_gained: float = 0.0
    resource_lost: float = 0.0
    resource_diversity_bonus: float = 0.0
    resource_hoarding_penalty: float = 0.0
    efficient_trade: float = 0.0
    port_usage: float = 0.0
    bank_trade_4to1: float = 0.0
    high_value_settlement: float = 0.0
    diverse_settlement: float = 0.0
    port_access_gained: float = 0.0
    robber_block_opponent: float = 0.0
    robber_steal_success: float = 0.0
    rank_1st: float = 0.0
    rank_2nd: float = 0.0
    rank_3rd: float = 0.0
    rank_4th: float = 0.0
    vp_lead_bonus: float = 0.0
    vp_behind_penalty: float = 0.0
    step_penalty: float = 0.0
    turn_penalty: float = 0.0
    second_place_reward: float = 0.0
    exploration_bonus: float = 0.0
    invalid_action_penalty: float = 0.0

    normalize_rewards: bool = False
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0

    scale_early_game: float = 0.0
    scale_mid_game: float = 0.0
    scale_late_game: float = 0.0

class StateEncoder:
    """
    Featurizes Catan game state into what I would consider good or bad features,
    little bit biased but shouldn't inhibit learning a decent policy.
    """
    
    def __init__(self, config=None):
        self.epsilon = 1e-8
        
        # standard board
        self.num_hexes = 19 
        self.num_vertices = 54
        self.num_edges = 72
        
        self.config = config
        
        # standard Catan role probs, generalize with combinatorial calc for variant num sides, num dice, etc.
        self.pip_values = {
            2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
            8: 5, 9: 4, 10: 3, 11: 2, 12: 1
        }
    
    def get_player_vector_dim(self, config) -> int:
        """Calculate player vector dimension based on config."""
        dim = 0
        
        dim += 5  # raw resources (wheat, wood, brick, sheep, ore)
        dim += 6  # resource metrics (total, diversity, settlement/city/road/dev potential)
        dim += 1  # conditional: trade_potential OR scarcity_value (always 1 feature)
        dim += 1  # robber vulnerability
        # 13
        
        dim += 6  # settlements, cities, roads, dev_ratio, production_rate, production_diversity
        if config.allow_bank_trading:
            dim += 1  # port efficiency
        dim += 5  # expansion_pressure, board_control, robber_pressure, connectivity, blocking
        # Subtotal: 11 or 12
        
        # === DEVELOPMENT & ACHIEVEMENTS ===
        dim += 10  # total_dev, playable_dev, knights, VPs, army/road progress, achievements
        # 10
        
        dim += 10  # VP metrics, gaps, win proximity, competitive position, hidden VP
        if config.allow_trading:
            dim += 1 # trade potential (again, separate from resources)
        dim += 4  # threat_level, game_phase (early, mid, late)
        # 14 or 15
        
        dim += 8  # can_afford (4), turns_until (2), flexibility, dev_card_value
        # 8
        
        num_opponents_to_analyze = min(2, config.num_players - 1)
        dim += num_opponents_to_analyze * 10 # 10 per opponent
        # 0, 10, or 20
        
        return dim
    
    def encode(self, game, player_id) -> Dict:
        """Encode game state with heavy featurization."""
        cache = self._build_cache(game, player_id)
        
        hex_feats = self._encode_hexes_strategic(game, player_id, cache)
        vertex_feats = self._encode_vertices_strategic(game, player_id, cache)
        edge_feats = self._encode_edges_strategic(game, player_id, cache)
        edge_indices, edge_types = self._build_connectivity(game)
        player_vec = self._encode_player_strategic(game, player_id, cache)
        global_ctx = self._encode_global_strategic(game, player_id, cache)
        action_masks = self._generate_action_masks(game, player_id)
        
        return {
            'hex_features': hex_feats,
            'vertex_features': vertex_feats,
            'edge_features': edge_feats,
            'graph_edges': edge_indices,
            'edge_types': edge_types,
            'player_vector': player_vec,
            'global_context': global_ctx,
            'action_masks': action_masks
        }

    def _build_cache(self, game, player_id) -> Dict:
        """Pre-compute expensive strategic calculations once."""
        cache = {
            'resource_scarcity': self._compute_resource_scarcity(game),
            'hex_control': self._compute_hex_control(game),
            'expansion_map': self._compute_expansion_map(game),
            'blocking_value': self._compute_blocking_values(game, player_id),
            'dev_card_value': self._compute_dev_card_value(game, player_id),
            'game_phase': self._compute_game_phase(game),
            'robber_pressure': self._compute_robber_pressure(game, player_id),
        }
        
        # Conditional: only compute trade potential if trading enabled
        if game.config.allow_bank_trading or game.config.allow_trading:
            cache['trade_potential'] = self._compute_trade_potential(game, player_id)
        else:
            cache['trade_potential'] = 0.0
        
        return cache

    def _encode_hexes_strategic(self, game, player_id, cache) -> np.ndarray:
        """[19, 12] hex features."""
        hex_features = []
        
        for hex_id in sorted(game.hexes.keys()):
            hex_obj = game.hexes[hex_id]
            resource_id = self._resource_to_id(hex_obj.resource)
            roll_number = hex_obj.roll_number if hex_obj.roll_number else 0
            pip_value = self.pip_values.get(roll_number, 0) / 5.0
            scarcity = cache['resource_scarcity'].get(hex_obj.resource, 0.5)
            
            # Count buildings on this hex
            our_buildings = enemy_buildings = total_buildings = 0
            for vid in hex_obj.adjacent_vertices:
                v = game.vertices[vid]
                if v.owner is not None:
                    total_buildings += 1
                    if v.owner == player_id:
                        our_buildings += 1
                    else:
                        enemy_buildings += 1
            
            our_control = our_buildings / 3.0
            enemy_control = enemy_buildings / 3.0
            saturation = total_buildings / 3.0
            has_robber = 1.0 if hex_obj.has_robber else 0.0
            robber_targets_us = 1.0 if has_robber and our_buildings > 0 else 0.0
            
            # Composite strategic value weighted by scarcity and penalized by robber
            strategic_value = pip_value * scarcity * our_control * (1.0 - has_robber * 0.5)
            is_contested = 1.0 if our_buildings > 0 and enemy_buildings > 0 else 0.0
            
            hex_features.append([
                resource_id, roll_number, pip_value, scarcity,
                our_control, enemy_control, saturation,
                has_robber, robber_targets_us, strategic_value,
                is_contested, pip_value * scarcity
            ])
        
        return np.array(hex_features, dtype=np.float32)

    def _encode_vertices_strategic(self, game, player_id, cache) -> np.ndarray:
        """[54, 16] vertex features."""
        vertex_features = []
        
        for vid in sorted(game.vertices.keys()):
            v = game.vertices[vid]
            owner_id = (v.owner + 1) if v.owner is not None else 0
            building_level = 0.0 if v.owner is None else (2.0 if v.is_city else 1.0)
            is_ours = 1.0 if v.owner == player_id else 0.0
            
            port_id = self._get_port_type_id(game, vid)
            has_port = 1.0 if port_id > 0 else 0.0
            port_value = self._compute_port_value(game, vid, player_id)
            
            # Aggregate production from adjacent hexes
            production_value = 0.0
            resource_counts = {}
            for hex_id in v.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.resource != 'desert' and hex_obj.roll_number:
                    pip = self.pip_values.get(hex_obj.roll_number, 0)
                    scarcity = cache['resource_scarcity'].get(hex_obj.resource, 0.5)
                    production_value += (pip / 5.0) * scarcity
                    resource_counts[hex_obj.resource] = resource_counts.get(hex_obj.resource, 0) + 1
            
            production_value = min(1.0, production_value / 3.0)
            production_diversity = len(resource_counts) / 3.0
            expansion_value = cache['expansion_map'].get(vid, 0.0)
            blocking_value = cache['blocking_value'].get(vid, 0.0)
            
            # Count friendly/enemy neighbors for threat analysis
            threat_level = friendly_support = 0.0
            for eid in v.adjacent_edges:
                edge = game.edges[eid]
                for adj_vid in {edge.vertex1, edge.vertex2}:
                    if adj_vid != vid:
                        adj_v = game.vertices[adj_vid]
                        if adj_v.owner is not None:
                            if adj_v.owner == player_id:
                                friendly_support += 1
                            else:
                                threat_level += 1
            
            threat_level = min(1.0, threat_level / 6.0)
            friendly_support = min(1.0, friendly_support / 6.0)
            
            # Composite value depends on ownership status
            if v.owner is None:
                build_value = (production_value * 0.4 + expansion_value * 0.3 + 
                            blocking_value * 0.2 + port_value * 0.1)
            else:
                build_value = production_value * 0.7 + port_value * 0.3
            
            vertex_features.append([
                owner_id, port_id, is_ours, building_level, has_port,
                production_value, production_diversity, port_value,
                expansion_value, blocking_value, threat_level, friendly_support,
                build_value,
                1.0 if v.owner is None else 0.0,
                1.0 if v.owner is not None and v.owner != player_id else 0.0,
                1.0 if production_value > 0.5 and v.owner is None else 0.0
            ])
        
        return np.array(vertex_features, dtype=np.float32)

    def _encode_edges_strategic(self, game, player_id, cache) -> np.ndarray:
        """[72, 8] edge features."""
        edge_features = []
        
        for eid in sorted(game.edges.keys()):
            e = game.edges[eid]
            owner_id = (e.owner + 1) if e.owner is not None else 0
            is_ours = 1.0 if e.owner == player_id else 0.0
            
            # Count connected roads/buildings
            our_road_connectivity = enemy_road_connectivity = 0
            for vid in {e.vertex1, e.vertex2}:
                v = game.vertices[vid]
                for adj_eid in v.adjacent_edges:
                    if adj_eid != eid:
                        adj_edge = game.edges[adj_eid]
                        if adj_edge.owner == player_id:
                            our_road_connectivity += 1
                        elif adj_edge.owner is not None:
                            enemy_road_connectivity += 1
                if v.owner == player_id:
                    our_road_connectivity += 2
            
            our_road_connectivity = min(1.0, our_road_connectivity / 8.0)
            enemy_road_connectivity = min(1.0, enemy_road_connectivity / 8.0)
            contributes_to_longest_road = 1.0 if (e.owner == player_id and 
                                                our_road_connectivity > 0.3) else 0.0
            
            # Average expansion value of endpoints
            expansion_value = sum(cache['expansion_map'].get(vid, 0.0) 
                                for vid in {e.vertex1, e.vertex2}) / 2.0
            
            # High blocking value if enemy buildings/roads nearby
            blocking_value = 0.0
            if e.owner is None:
                for vid in {e.vertex1, e.vertex2}:
                    v = game.vertices[vid]
                    if v.owner is not None and v.owner != player_id:
                        blocking_value = 0.7
                        break
                    for adj_eid in v.adjacent_edges:
                        if adj_eid != eid and game.edges[adj_eid].owner is not None:
                            if game.edges[adj_eid].owner != player_id:
                                blocking_value = max(blocking_value, 0.4)
            
            edge_features.append([
                owner_id, is_ours, our_road_connectivity, enemy_road_connectivity,
                contributes_to_longest_road, expansion_value, blocking_value,
                1.0 if e.owner is None else 0.0
            ])
        
        return np.array(edge_features, dtype=np.float32)

    def _encode_player_strategic(self, game, player_id, cache) -> np.ndarray:
        """Dynamic player vector adapting to config (num_players, trading enabled)."""
        features = []
        
        inv = game.get_player_inventory(player_id)
        dev = game.get_player_dev_cards(player_id)
        settlements, cities = game.get_buildings(player_id)
        my_vp = game.get_player_score(player_id)
        total_resources = sum(inv.values())
        
        features.extend([
            inv['wheat'] / 10.0, inv['wood'] / 10.0, inv['brick'] / 10.0,
            inv['sheep'] / 10.0, inv['ore'] / 10.0,
            total_resources / 20.0,
            self._resource_diversity_entropy(inv),
            min(inv['wood'], inv['brick'], inv['sheep'], inv['wheat']) / 5.0,
            min(inv['wheat'] // 2, inv['ore'] // 3) / 2.0,
            min(inv['wood'], inv['brick']) / 10.0,
            min(inv['wheat'], inv['sheep'], inv['ore']) / 3.0,
        ])
        
        if game.config.allow_bank_trading or game.config.allow_trading:
            features.append(cache['trade_potential'] / 10.0)
        else:
            features.append(sum(inv[r] * cache['resource_scarcity'].get(r, 0.5) 
                            for r in inv) / 20.0)
        
        features.append(1.0 if total_resources >= 8 else 0.0)
        
        features.extend([
            len(settlements) / 5.0,
            len(cities) / 4.0,
            len(game.players[player_id].owned_roads) / 15.0,
            len(cities) / max(1, len(settlements) + len(cities)),
            self._compute_production_rate(game, player_id) / 5.0,
            self._compute_production_diversity(game, player_id) / 5.0,
        ])
        
        if game.config.allow_bank_trading:
            features.append(self._compute_port_efficiency(game, player_id))
        
        features.extend([
            1.0 - (len(settlements) + len(cities)) / 9.0,
            self._compute_board_control(game, player_id),
            cache['robber_pressure'],
            game._longest_road_cache.get(player_id, 0) / 15.0,
            sum(cache['blocking_value'].values()) / 54.0
        ])
        
        total_dev = sum(dev.values())
        playable_dev = len(game.players[player_id]._get_playable_dev_cards(game))
        
        features.extend([
            total_dev / 10.0, playable_dev / 5.0,
            dev['knight'] / 5.0, dev['victory_point'] / 5.0,
            game.knights_played[player_id] / 10.0,
            (game.knights_played[player_id] / max(1, sum(game.knights_played.values()))),
            1.0 if game.largest_army_info[0] == player_id else 0.0,
            1.0 if game.longest_road_info[0] == player_id else 0.0,
            min(1.0, game.knights_played[player_id] / 3.0),
            min(1.0, game._longest_road_cache.get(player_id, 0) / 5.0)
        ])

        all_vps = [game.get_player_score(p) for p in range(game.config.num_players)]
        visible_vp = len(settlements) + 2 * len(cities)
        hidden_vp = my_vp - visible_vp
        
        features.extend([
            my_vp / 10.0, max(all_vps) / 10.0,
            (my_vp - max(all_vps)) / 10.0,
            (my_vp - np.mean(all_vps)) / 10.0,
            (my_vp / game.config.victory_points_to_win),
            1.0 if my_vp >= game.config.victory_points_to_win - 2 else 0.0,
            1.0 if my_vp == max(all_vps) else 0.0,
            sum(1 for vp in all_vps if vp >= my_vp) / game.config.num_players,
            sum(1 for vp in all_vps if vp >= my_vp - 2) / game.config.num_players,
            hidden_vp / 10.0,
        ])
        
        if game.config.allow_trading:
            features.append(cache['trade_potential'] / 10.0)
        
        features.extend([
            1.0 if my_vp == max(all_vps) and my_vp >= 7 else 0.0,
            cache['game_phase']['early'],
            cache['game_phase']['mid'],
            cache['game_phase']['late']
        ])
        
        features.extend([
            1.0 if game.can_afford(player_id, 'settlement') else 0.0,
            1.0 if game.can_afford(player_id, 'city') else 0.0,
            1.0 if game.can_afford(player_id, 'road') else 0.0,
            1.0 if game.can_afford(player_id, 'development_card') else 0.0,
            self._turns_until_affordable(game, player_id, 'settlement'),
            self._turns_until_affordable(game, player_id, 'city'),
            sum([game.can_afford(player_id, 'settlement'),
                game.can_afford(player_id, 'city'),
                game.can_afford(player_id, 'road'),
                game.can_afford(player_id, 'development_card')]) / 4.0,
            cache['dev_card_value']
        ])
        
        all_opps = [i for i in range(game.config.num_players) if i != player_id] # haha
        sorted_opps = sorted(all_opps, key=lambda i: game.get_player_score(i), reverse=True)
        num_opponents_to_analyze = min(2, len(all_opps))
        
        for idx in range(num_opponents_to_analyze):
            opp_id = sorted_opps[idx]
            opp_vp = game.get_player_score(opp_id)
            opp_inv = game.get_player_inventory(opp_id)
            opp_s, opp_c = game.get_buildings(opp_id)
            
            features.extend([
                opp_vp / 10.0, (opp_vp - my_vp) / 10.0,
                sum(opp_inv.values()) / 20.0,
                len(opp_s) / 5.0, len(opp_c) / 4.0,
                game.knights_played[opp_id] / 10.0,
                1.0 if opp_vp >= game.config.victory_points_to_win - 2 else 0.0,
                1.0 if game.largest_army_info[0] == opp_id else 0.0,
                1.0 if game.longest_road_info[0] == opp_id else 0.0,
                (opp_id - player_id) / game.config.num_players
            ])
        
        return np.array(features, dtype=np.float32)

    def _encode_global_strategic(self, game, player_id, cache) -> np.ndarray:
        """[15] global strategic context."""
        all_vps = [game.get_player_score(p) for p in range(game.config.num_players)]
        
        features = [
            game.turn / 100.0,
            1.0 if game.current_player == player_id else 0.0,
            len(game.dev_card_deck) / 25.0,
            1.0 if len(game.dev_card_deck) == 0 else 0.0,
            cache['game_phase']['early'],
            cache['game_phase']['mid'],
            cache['game_phase']['late'],
            max(all_vps) / 10.0,
            min(all_vps) / 10.0,
            (max(all_vps) - min(all_vps)) / 10.0,
            np.std(all_vps) / 10.0,
            1.0 if max(all_vps) >= game.config.victory_points_to_win - 3 else 0.0,
            max(all_vps) / game.config.victory_points_to_win,
            sum(sum(game.get_player_inventory(p).values()) 
                for p in range(game.config.num_players)) / 80.0,
            sum(len(game.get_buildings(p)[0]) + len(game.get_buildings(p)[1]) 
                for p in range(game.config.num_players)) / 36.0
        ]
        
        return np.array(features, dtype=np.float32)

    def _compute_resource_scarcity(self, game) -> Dict[str, float]:
        """Inverse pip frequency (rarer resources = higher scarcity)."""
        resource_pips = {'wheat': 0, 'wood': 0, 'brick': 0, 'sheep': 0, 'ore': 0, 'desert': 0}
        
        for hex_obj in game.hexes.values():
            if hex_obj.resource != 'desert' and hex_obj.roll_number:
                pip = self.pip_values.get(hex_obj.roll_number, 0)
                resource_pips[hex_obj.resource] += pip
        
        total_pips = sum(resource_pips.values())
        if total_pips > 0:
            scarcity = {res: 1.0 - (resource_pips[res] / total_pips) 
                    for res in resource_pips if res != 'desert'}
            scarcity['desert'] = 0.0
        else:
            scarcity = {r: 0.5 for r in resource_pips}
        
        return scarcity

    def _compute_hex_control(self, game) -> Dict[int, List[int]]:
        """Building count per player on each hex."""
        control = {}
        for hex_id, hex_obj in game.hexes.items():
            player_counts = {i: 0 for i in range(game.config.num_players)}
            for vid in hex_obj.adjacent_vertices:
                v = game.vertices[vid]
                if v.owner is not None:
                    player_counts[v.owner] += 1
            control[hex_id] = player_counts
        return control

    def _compute_expansion_map(self, game) -> Dict[int, float]:
        """How open is each unoccupied vertex for expansion?"""
        expansion = {}
        for vid in range(self.num_vertices):
            v = game.vertices[vid]
            if v.owner is not None:
                expansion[vid] = 0.0
                continue
            
            empty_adjacent = sum(1 for eid in v.adjacent_edges
                                for adj_vid in {game.edges[eid].vertex1, game.edges[eid].vertex2}
                                if adj_vid != vid and game.vertices[adj_vid].owner is None)
            expansion[vid] = min(1.0, empty_adjacent / 4.0)
        
        return expansion

    def _compute_blocking_values(self, game, player_id) -> Dict[int, float]:
        """How much enemy pressure exists at each unoccupied vertex?"""
        blocking = {}
        for vid in range(self.num_vertices):
            v = game.vertices[vid]
            if v.owner is not None:
                blocking[vid] = 0.0
                continue
            
            enemy_pressure = sum(1 for eid in v.adjacent_edges
                                for adj_vid in {game.edges[eid].vertex1, game.edges[eid].vertex2}
                                if adj_vid != vid and game.vertices[adj_vid].owner is not None 
                                and game.vertices[adj_vid].owner != player_id)
            blocking[vid] = min(1.0, enemy_pressure / 3.0)
        
        return blocking

    def _compute_trade_potential(self, game, player_id) -> float:
        """0-10 scale combining trade ratio quality and resource quantity."""
        inv = game.get_player_inventory(player_id)
        
        if game.config.allow_bank_trading:
            ratios = game.get_player_trade_ratios(player_id)
            best_ratio = min(ratios.values())
            ratio_quality = (4 - best_ratio) / 2.0
        else:
            ratio_quality = 0.0
        
        resource_quantity = sum(inv.values()) / 10.0
        return min(10.0, ratio_quality * 5 + resource_quantity)

    def _compute_dev_card_value(self, game, player_id) -> float:
        """Weighted value: catching up + endgame push + availability."""
        my_vp = game.get_player_score(player_id)
        max_vp = max(game.get_player_score(p) for p in range(game.config.num_players))
        
        behind_penalty = 1.0 - (my_vp / max(1, max_vp))
        endgame_bonus = my_vp / game.config.victory_points_to_win
        deck_availability = len(game.dev_card_deck) / 25.0
        
        return (behind_penalty * 0.4 + endgame_bonus * 0.4 + deck_availability * 0.2)

    def _compute_game_phase(self, game) -> Dict[str, float]:
        """Early/mid/late game based on buildings and turns."""
        total_buildings = sum(len(game.get_buildings(p)[0]) + len(game.get_buildings(p)[1])
                            for p in range(game.config.num_players))
        saturation = total_buildings / 36
        turn = game.turn
        
        early = max(0.0, 1.0 - turn / 15.0) * max(0.0, 1.0 - saturation * 2)
        late = min(1.0, max(saturation * 1.5 - 0.5, (turn - 40) / 20.0))
        mid = max(0.0, 1.0 - early - late)
        
        return {'early': early, 'mid': mid, 'late': late}

    def _compute_robber_pressure(self, game, player_id) -> float:
        """Expected production loss from robber on our buildings."""
        if game.robber_hex_id is None:
            return 0.0
        
        hex_obj = game.hexes[game.robber_hex_id]
        our_buildings = sum(1 for vid in hex_obj.adjacent_vertices 
                        if game.vertices[vid].owner == player_id)
        
        if our_buildings == 0:
            return 0.0
        
        pip = self.pip_values.get(hex_obj.roll_number, 0) / 5.0
        return min(1.0, our_buildings * pip / 3.0)

    def _compute_production_rate(self, game, player_id) -> float:
        """Expected resources per turn (probability-weighted)."""
        rate = 0.0
        settlements, cities = game.get_buildings(player_id)
        
        for v in settlements:
            for hex_id in v.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.resource != 'desert' and hex_obj.roll_number and not hex_obj.has_robber:
                    pip = self.pip_values.get(hex_obj.roll_number, 0)
                    rate += pip / 36.0
        
        for v in cities:
            for hex_id in v.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.resource != 'desert' and hex_obj.roll_number and not hex_obj.has_robber:
                    pip = self.pip_values.get(hex_obj.roll_number, 0)
                    rate += 2 * pip / 36.0
        
        return rate

    def _compute_production_diversity(self, game, player_id) -> float:
        """Count of unique resource types produced."""
        settlements, cities = game.get_buildings(player_id)
        resources = set()
        
        for v in settlements + cities:
            for hex_id in v.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.resource != 'desert':
                    resources.add(hex_obj.resource)
        
        return float(len(resources))

    def _compute_port_efficiency(self, game, player_id) -> float:
        """Normalized trade ratio quality (4:1 = 0.0, 2:1 = 1.0)."""
        ratios = game.get_player_trade_ratios(player_id)
        best_ratio = min(ratios.values())
        return (4 - best_ratio) / 2.0

    def _compute_board_control(self, game, player_id) -> float:
        """Fraction of hexes we touch with buildings."""
        settlements, cities = game.get_buildings(player_id)
        our_hexes = set()
        for v in settlements + cities:
            for hex_id in v.adjacent_hexes:
                our_hexes.add(hex_id)
        return len(our_hexes) / 19.0

    def _resource_diversity_entropy(self, inv: Dict[str, int]) -> float:
        """Shannon entropy normalized to [0,1]."""
        total = sum(inv.values())
        if total == 0:
            return 0.0
        probs = [count / total for count in inv.values() if count > 0]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        return entropy / np.log2(5)

    def _turns_until_affordable(self, game, player_id, item_type: str) -> float:
        """Estimated turns to afford item based on production rate."""
        inv = game.get_player_inventory(player_id)
        production_rate = self._compute_production_rate(game, player_id)
        
        if game.can_afford(player_id, item_type):
            return 0.0
        
        costs = {
            'settlement': {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1},
            'city': {'wheat': 2, 'ore': 3},
            'road': {'wood': 1, 'brick': 1},
            'development_card': {'wheat': 1, 'sheep': 1, 'ore': 1}
        }
        
        cost = costs.get(item_type, {})
        needed = sum(max(0, cost.get(r, 0) - inv[r]) for r in ['wheat', 'wood', 'brick', 'sheep', 'ore'])
        
        if production_rate < 0.1:
            return 1.0
        
        turns = needed / max(0.1, production_rate)
        return min(1.0, turns / 5.0)

    def _compute_port_value(self, game, vertex_id, player_id) -> float:
        """Strategic value weighted by inventory."""
        port_id = self._get_port_type_id(game, vertex_id)
        if port_id == 0:
            return 0.0
        
        inv = game.get_player_inventory(player_id)
        
        if port_id == 1: # 3:1
            return 0.5
        
        # 2:1 special
        port_resources = {2: 'wheat', 3: 'wood', 4: 'brick', 5: 'sheep', 6: 'ore'}
        resource = port_resources.get(port_id)
        return 0.8 if resource and inv[resource] >= 3 else 0.3

    def _resource_to_id(self, resource: str) -> int:
        mapping = {'wheat': 0, 'wood': 1, 'brick': 2, 'sheep': 3, 'ore': 4, 'desert': 5}
        return mapping[resource]

    def _get_port_type_id(self, game, vertex_id: int) -> int:
        for edge_id in game.vertices[vertex_id].adjacent_edges:
            if edge_id in game.ports:
                port = game.ports[edge_id]
                if port['resource'] is None:
                    return 1  # 3:1
                mapping = {'wheat': 2, 'wood': 3, 'brick': 4, 'sheep': 5, 'ore': 6}
                return mapping[port['resource']]
        return 0

    def _build_connectivity(self, game) -> Tuple[np.ndarray, np.ndarray]:
        """Build heterogeneous graph edges: hex-vertex, vertex-edge, vertex-vertex."""
        edge_list = []
        edge_types = []
        
        hex_offset = 0
        vertex_offset = self.num_hexes
        edge_offset = self.num_hexes + self.num_vertices
        
        # tile-vertex connections
        for hex_id in sorted(game.hexes.keys()):
            hex_idx = hex_offset + hex_id
            for vid in game.hexes[hex_id].adjacent_vertices:
                vert_idx = vertex_offset + vid
                edge_list.extend([[hex_idx, vert_idx], [vert_idx, hex_idx]])
                edge_types.extend([0, 0])
        
        # vertex-edge connections
        for vid in sorted(game.vertices.keys()):
            vert_idx = vertex_offset + vid
            for eid in game.vertices[vid].adjacent_edges:
                edge_idx = edge_offset + eid
                edge_list.extend([[vert_idx, edge_idx], [edge_idx, vert_idx]])
                edge_types.extend([1, 1])
        
        # vertex-vertex via edges
        for eid in sorted(game.edges.keys()):
            edge = game.edges[eid]
            v1_idx = vertex_offset + edge.vertex1
            v2_idx = vertex_offset + edge.vertex2
            edge_list.extend([[v1_idx, v2_idx], [v2_idx, v1_idx]])
            edge_types.extend([2, 2])
        
        edge_array = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        edge_types = np.array(edge_types, dtype=np.int64)
        
        return edge_array, edge_types

    def _generate_action_masks(self, game, player_id) -> Dict:
        """Validity masks for all action types."""
        masks = {}
        
        settlement_mask = np.zeros(self.num_vertices, dtype=np.float32)
        city_mask = np.zeros(self.num_vertices, dtype=np.float32)
        road_mask = np.zeros(self.num_edges, dtype=np.float32)
        
        if game.can_afford(player_id, 'settlement'):
            for vid in game.get_valid_settlement_locations(player_id):
                settlement_mask[vid] = 1.0
        
        if game.can_afford(player_id, 'city'):
            settlements, _ = game.get_buildings(player_id)
            for v in settlements:
                if v.id in game.get_valid_city_locations(player_id):
                    city_mask[v.id] = 1.0
        
        if game.can_afford(player_id, 'road'):
            for eid in game.get_valid_road_locations(player_id):
                road_mask[eid] = 1.0
        
        masks['settlement'] = settlement_mask
        masks['city'] = city_mask
        masks['road'] = road_mask
        
        # Development card masks
        card_mask = np.zeros(5, dtype=np.float32)
        if game.can_afford(player_id, 'development_card') and len(game.dev_card_deck) > 0:
            card_mask[0] = 1.0
        
        playable = game.players[player_id]._get_playable_dev_cards(game)
        if 'knight' in playable:
            card_mask[1] = 1.0
        if 'road_building' in playable:
            card_mask[2] = 1.0
        if 'year_of_plenty' in playable:
            card_mask[3] = 1.0
        if 'monopoly' in playable:
            card_mask[4] = 1.0
        
        masks['card'] = card_mask
        
        return masks

class CatanGNNTransformerNetwork(nn.Module):
    class CatanGNNTransformerNetwork(nn.Module):
        """
        A deep neural network architecture designed for the game of Catan, utilizing
        Graph Neural Networks (GNNs) to process the board state and a Transformer
        to fuse all information for policy and value prediction.

        The architecture is a hybrid model structured to efficiently handle the
        multi-relational graph structure of the Catan board.

        Architecture Components & Mathematical Details:

        1. GNN Branches for Graph Components:
            * **Hexes (19 nodes):** Nodes representing resource tiles (hexagons).
                -   **Input:** Concatenation of embeddings for resource type (6 categories)
                    and dice roll number (13 categories), plus continuous features (e.g.,
                    robber presence, resource probabilities).
                -   **GNN:** Uses $L$ layers of $\text{GATConv}$, where $L=\text{num\_gnn\_layers}$.
                    The GATConv uses a single attention head ($H=1$) and no concatenation.
                    The core operation involves updating node features $h_i$ by aggregating
                    information from neighbors $j \in \mathcal{N}(i)$:
                    $$h_i' = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij} W h_j \right)$$
                    where $\alpha_{ij}$ is the attention coefficient calculated via an
                    attentional mechanism on $W h_i$ and $W h_j$.
                -   **Output:** $19 \times \text{hidden\_dim}$ feature matrix.
            * **Vertices (54 nodes):** Nodes representing potential/existing settlement/city locations.
                -   **Input:** Embeddings for owner ID (up to 5 categories) and port type
                    (7 categories), plus continuous features (e.g., building level, longest
                    road/largest army contributions).
                -   **GNN:** $L$ layers of $\text{GATConv}$ ($H=1$, no concat) with similar
                    mathematical structure to the Hex GNN.
                -   **Output:** $54 \times \text{hidden\_dim}$ feature matrix.
            * **Edges (72 nodes):** Nodes representing potential/existing road locations.
                -   **Input:** Embedding for owner ID (up to 5 categories) plus continuous
                    features (e.g., longest road contribution).
                -   **GNN:** $L$ layers of $\text{GATConv}$ ($H=1$, no concat).
                -   **Output:** $72 \times \text{hidden\_dim}$ feature matrix.

        2. MLP Branches for Non-Graph Context:
            * **Player Stats (1 node):** Processes the features specific to the current player.
                -   **MLP:** $\text{Linear} \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout} \to \text{Linear} \to \text{LayerNorm} \to \text{ReLU}$.
                -   **Reasoning:** Player stats (resources, victory points, development cards)
                    are non-spatial features requiring a standard feed-forward network to map
                    them into the $\text{hidden\_dim}$ space for fusion with graph features.
                -   **Output:** $1 \times \text{hidden\_dim}$ feature vector.
            * **Global Context (1 node):** Processes features shared across the board (e.g.,
                turn number, current dice roll, development card piles).
                -   **MLP:** $\text{Linear} \to \text{LayerNorm} \to \text{ReLU} \to \text{Linear} \to \text{LayerNorm} \to \text{ReLU}$.
                -   **Reasoning:** Similar to Player Stats, these non-spatial features are
                    mapped into the $\text{hidden\_dim}$ space.
                -   **Output:** $1 \times \text{hidden\_dim}$ feature vector.

        3. Single-Layer Transformer for Information Fusion:
            * **Input Sequence:** The outputs from all branches are concatenated into a single
                sequence of $19 + 54 + 72 + 1 + 1 = 147$ tokens (sequences of length $\text{max\_seq\_len}$)
                with dimension $\text{hidden\_dim}$.
            * **Positional Embedding:** A learned positional embedding is added to the input
                sequence:
                $$E_{\text{in}} = \text{Concat}(\text{Hex}_{\text{out}}, \dots, \text{Global}_{\text{out}}) + \text{PosEmb}$$
            * **Transformer Encoder:** A single layer ($\text{num\_layers}=1$) $\text{TransformerEncoder}$
                with $\text{num\_attention\_heads}$ is applied. This layer performs **Self-Attention**
                across all feature tokens:
                $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
                where $Q, K, V$ are the Query, Key, and Value matrices derived from the input
                sequence, and $d_k = \text{hidden\_dim} / \text{num\_attention\_heads}$.
            * **Reasoning:** The Transformer is crucial for enabling **cross-modality interaction**.
                It allows a hex feature vector to directly attend to a player's resource count
                or a vertex's building status, creating a rich, globally aware representation.
                This is key for complex, long-range dependencies in Catan.
            * **Output:** $147 \times \text{hidden\_dim}$ fused feature matrix.

        4. Shared Trunk and Policy/Value Outputs:
            * **Pooling:** The final feature vector for the **Global Context** token (the last
                token in the sequence, $\text{transformer\_out}[:, -1]$) is typically used
                as a **pooled** representation of the entire state.
            * **Trunk:** This pooled feature is passed through a shared MLP trunk:
                $$\text{Trunk}_{\text{out}} = \text{ReLU}(\text{Linear}(\text{ReLU}(\text{Linear}(\text{pooled}))))) $$
            * **Policy Heads:** A $\text{Multi-Head}$ architecture is used to predict logits
                for various action types (e.g., 'settlement', 'road', 'trade').
                $$\text{Logits}_{\text{type}} = \text{Linear}(\text{ReLU}(\text{Linear}(\text{Trunk}_{\text{out}})))$$
                Final output is typically passed through $\text{softmax}$ by the training loop.
            * **Value Head:** A separate head predicts the scalar value of the state.
                $$\text{Value} = \text{Tanh}(\text{Linear}(\text{ReLU}(\text{Trunk}_{\text{out}}))) \times \text{value\_scale}$$
                The $\text{Tanh}$ activation bounds the raw prediction between $[-1, 1]$, and
                $\text{value\_scale}$ scales it to the desired range (e.g., victory points or win probability).
            * **Reasoning:** Separating the output into multiple policy heads allows the
                network to focus on the sub-task relevant to the current game state, improving
                sample efficiency and generalization. Using a **shared trunk** ensures that
                the high-level features learned benefit both the policy and value functions
                (a common practice in reinforcement learning).
        """
    
    def __init__(
        self,
        config,

        hidden_dim=48,
        num_gnn_layers=1,
        num_attention_heads=2,
        dropout=0.1,
        
        resource_emb_dim=6,
        roll_emb_dim=4,
        owner_emb_dim=4,
        port_emb_dim=4,
        
        player_mlp_hidden=96,
        global_mlp_hidden=24,
        
        trunk_hidden=192,

        policy_hidden_1=96,
        policy_hidden_2=48,
        
        value_hidden=96,
        value_scale=10.0,
        
        init_policy_gain=0.1,
        init_value_gain=1.0
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.value_scale = value_scale
        
        self.game_board = CatanGame(
            config, 
            players=[RandomAgent(i, 'color') for i in range(config.num_players)]
        )
        
        # adjacency dict for hexes
        hex_adj = {h.id: h.adjacent_hexes for h in self.game_board.hexes.values()}
        hex_edges = self._build_single_graph_edge_index(hex_adj, 19)
        self.register_buffer('hex_edge_template', hex_edges)
        
        # adjacency dict for vertices
        vertex_adj = {v.id: v.adjacent_vertices for v in self.game_board.vertices.values()}
        vertex_edges = self._build_single_graph_edge_index(vertex_adj, 54)
        self.register_buffer('vertex_edge_template', vertex_edges)
        
        # adjacency dict for edges
        edge_adj = {e.id: e.adjacent_edges for e in self.game_board.edges.values()}
        edge_edges = self._build_single_graph_edge_index(edge_adj, 72)
        self.register_buffer('edge_edge_template', edge_edges)
        
        encoder = StateEncoder(config)
        self.player_stat_dim = encoder.get_player_vector_dim(config)
        self.global_feat_dim = 15
        
        self.max_seq_len = 19 + 54 + 72 + 1 + 1  # hexes + vertices + edges + player + global
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_seq_len, hidden_dim) * 0.02
        )
        
        # ===== EMBEDDING LAYERS =====
        self.resource_emb = nn.Embedding(6, resource_emb_dim)
        self.roll_emb = nn.Embedding(13, roll_emb_dim)
        self.owner_emb = nn.Embedding(max(5, config.num_players + 1), owner_emb_dim)
        self.port_emb = nn.Embedding(7, port_emb_dim)
        
        # Hex GNN
        hex_input_dim = resource_emb_dim + roll_emb_dim + 10  # embeddings + continuous features
        self.hex_gnn_layers = nn.ModuleList([
            GATConv(hex_input_dim if i == 0 else hidden_dim, 
                   hidden_dim, heads=1, concat=False, dropout=dropout)
            for i in range(num_gnn_layers)
        ])
        self.hex_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])
        
        # Vertex GNN
        vertex_input_dim = owner_emb_dim + port_emb_dim + 14
        self.vertex_gnn_layers = nn.ModuleList([
            GATConv(vertex_input_dim if i == 0 else hidden_dim,
                   hidden_dim, heads=1, concat=False, dropout=dropout)
            for i in range(num_gnn_layers)
        ])
        self.vertex_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])
        
        # Edge GNN
        edge_input_dim = owner_emb_dim + 7
        self.edge_gnn_layers = nn.ModuleList([
            GATConv(edge_input_dim if i == 0 else hidden_dim,
                   hidden_dim, heads=1, concat=False, dropout=dropout)
            for i in range(num_gnn_layers)
        ])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])
        
        # Player stats MLP
        self.player_mlp = nn.Sequential(
            nn.Linear(self.player_stat_dim, player_mlp_hidden),
            nn.LayerNorm(player_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(player_mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_feat_dim, global_mlp_hidden),
            nn.LayerNorm(global_mlp_hidden),
            nn.ReLU(),
            nn.Linear(global_mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.ReLU()
        )

        def make_head(output_dim):
            """Create a policy head"""
            return nn.Sequential(
                nn.Linear(trunk_hidden, policy_hidden_1),
                nn.LayerNorm(policy_hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(policy_hidden_1, policy_hidden_2),
                nn.LayerNorm(policy_hidden_2),
                nn.ReLU(),
                nn.Linear(policy_hidden_2, output_dim)
            )
        
        self.heads = nn.ModuleDict({
            'action': make_head(8),
            'settlement': make_head(54),
            'city': make_head(54),
            'road': make_head(72),
            'tile': make_head(19),
            'steal_player': make_head(config.num_players),
            'dev_card': make_head(5),
            'resource': make_head(5),
            'resource2': make_head(5),
            'trade_player': make_head(config.num_players),
            'trade_give_resource': make_head(6),
            'trade_receive_resource': make_head(6),
            'accept_trade': make_head(2),
            'exchange_give': make_head(5),
            'exchange_receive': make_head(5),
            'discard': make_head(5),
        })
        
        self.value_head = nn.Sequential(
            nn.Linear(trunk_hidden, value_hidden),
            nn.LayerNorm(value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh()
        )

        self._init_weights(init_policy_gain, init_value_gain)
    
    def _build_single_graph_edge_index(self, adj_dict, num_nodes):
        """Build edge_index for single graph"""
        src, dst = [], []
        for src_id, neighbors in adj_dict.items():
            for dst_id in neighbors:
                src.append(src_id)
                dst.append(dst_id)
        
        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        
        return torch.tensor([src, dst], dtype=torch.long)
    
    def _batch_edge_index(self, template, batch_size, nodes_per_graph):
        """Expand template to batch"""
        device = template.device
        offsets = torch.arange(batch_size, device=device) * nodes_per_graph
        batched = template.unsqueeze(0).repeat(batch_size, 1, 1)
        batched = batched + offsets.view(-1, 1, 1)
        return batched.transpose(0, 1).reshape(2, -1)
    
    def _create_hex_edges(self, batch_size, device):
        return self._batch_edge_index(self.hex_edge_template, batch_size, 19)
    
    def _create_vertex_edges(self, batch_size, device):
        return self._batch_edge_index(self.vertex_edge_template, batch_size, 54)
    
    def _create_edge_edges(self, batch_size, device):
        return self._batch_edge_index(self.edge_edge_template, batch_size, 72)
    
    def _init_weights(self, policy_gain, value_gain):
        """Initialize weights with small values for stable training"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'heads' in name:
                    nn.init.orthogonal_(param, gain=policy_gain)
                elif 'value_head' in name:
                    nn.init.orthogonal_(param, gain=value_gain)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state_dict, head_type='action', mask=None):
        # === 1-5. Process graph (unchanged) ===
        hex_out = self._process_hexes(state_dict)
        vertex_out = self._process_vertices(state_dict)
        edge_out = self._process_edges(state_dict)
        player_out = self.player_mlp(state_dict['player_vector'])
        global_out = self.global_mlp(state_dict['global_context'])
        
        transformer_input = torch.cat([
            hex_out, vertex_out, edge_out,
            player_out.unsqueeze(1),
            global_out.unsqueeze(1)
        ], dim=1)
        
        seq_len = transformer_input.size(1)
        transformer_input = transformer_input + self.pos_embedding[:, :seq_len, :]

        transformer_out = self.transformer(transformer_input)
        
        pooled = transformer_out[:, -1]
        
        trunk_out = self.trunk(pooled)

        logits = self.heads[head_type](trunk_out)
        value = self.value_head(trunk_out).squeeze(-1) * self.value_scale
        
        return logits, value
    
    def _process_hexes(self, state_dict):
        """Process hex nodes with GNN"""
        batch_size = state_dict['hex_features'].shape[0]
        hex_feats = state_dict['hex_features']
        
        resource_ids = hex_feats[:, :, 0].long()
        roll_ids = hex_feats[:, :, 1].long()
        continuous = hex_feats[:, :, 2:]
        
        resource_emb = self.resource_emb(resource_ids)
        roll_emb = self.roll_emb(roll_ids)
        
        x = torch.cat([resource_emb, roll_emb, continuous], dim=-1)
        x = x.view(batch_size * 19, -1)
        
        edge_index = self._create_hex_edges(batch_size, x.device)
        
        for i, (gnn, norm) in enumerate(zip(self.hex_gnn_layers, self.hex_norms)):
            x_new = gnn(x, edge_index)
            x_new = norm(x_new)
            x = F.relu(x_new)
            x = torch.clamp(x, -10.0, 10.0)
            if i < len(self.hex_gnn_layers) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
        
        x = x.view(batch_size, 19, self.hidden_dim)
        return x
    
    def _process_vertices(self, state_dict):
        """Process vertex nodes with GNN"""
        batch_size = state_dict['vertex_features'].shape[0]
        vertex_feats = state_dict['vertex_features']
        
        owner_ids = vertex_feats[:, :, 0].long()
        port_ids = vertex_feats[:, :, 1].long()
        continuous = vertex_feats[:, :, 2:]
        
        owner_emb = self.owner_emb(owner_ids)
        port_emb = self.port_emb(port_ids)
        
        x = torch.cat([owner_emb, port_emb, continuous], dim=-1)
        x = x.view(batch_size * 54, -1)
        
        edge_index = self._create_vertex_edges(batch_size, x.device)
        
        for i, (gnn, norm) in enumerate(zip(self.vertex_gnn_layers, self.vertex_norms)):
            x_new = gnn(x, edge_index)
            x_new = norm(x_new)
            x = F.relu(x_new)
            x = torch.clamp(x, -10.0, 10.0)
            if i < len(self.vertex_gnn_layers) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
        
        x = x.view(batch_size, 54, self.hidden_dim)
        return x
    
    def _process_edges(self, state_dict):
        """Process edge nodes with GNN"""
        batch_size = state_dict['edge_features'].shape[0]
        edge_feats = state_dict['edge_features']
        
        owner_ids = edge_feats[:, :, 0].long()
        continuous = edge_feats[:, :, 1:]
        
        owner_emb = self.owner_emb(owner_ids)
        
        x = torch.cat([owner_emb, continuous], dim=-1)
        x = x.view(batch_size * 72, -1)
        
        edge_index = self._create_edge_edges(batch_size, x.device)
        
        for i, (gnn, norm) in enumerate(zip(self.edge_gnn_layers, self.edge_norms)):
            x_new = gnn(x, edge_index)
            x_new = norm(x_new)
            x = F.relu(x_new)
            x = torch.clamp(x, -10.0, 10.0)
            if i < len(self.edge_gnn_layers) - 1:
                x = F.dropout(x, p=0.1, training=self.training)
        
        x = x.view(batch_size, 72, self.hidden_dim)
        return x
    
    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self):
        """Get detailed parameter breakdown by module"""
        breakdown = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                module_name = name.split('.')[0]
                if module_name not in breakdown:
                    breakdown[module_name] = 0
                breakdown[module_name] += param.numel()
        return breakdown

class CatanRLAgent(Player):
    def __init__(self, id: int, color: str, config, network: CatanGNNTransformerNetwork = None, temperature=1.0):
        super().__init__(id, color)
        
        self.config = config
        self.temperature = temperature
        self.encoder = StateEncoder(config)
        
        if network is None:
            network = CatanGNNTransformerNetwork(config)

        self.network = network
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.network.eval()
        
        for param in self.network.parameters():
            param.requires_grad = False

        self.allow_trading = config.allow_trading
        self.allow_bank_trading = config.allow_bank_trading
        
        self.profile_times = {
            'encode': [], 'mask': [], 'forward': [], 'sample': []
        }
    
    def _encode_state(self, game) -> Dict:
        """Convert game state to network input tensors."""
        start = time.time()
        state_dict = self.encoder.encode(game, self.id)
        
        tensor_dict = {}
        for key, value in state_dict.items():
            if key == 'action_masks':
                tensor_dict[key] = {
                    k: torch.tensor(v, dtype=torch.float32, device=self.device)
                    for k, v in value.items()
                }
            elif key in ['graph_edges', 'edge_types']:
                tensor_dict[key] = torch.tensor(value, dtype=torch.long, device=self.device)
            else:
                tensor_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        for key in ['hex_features', 'vertex_features', 'edge_features', 
                   'player_vector', 'global_context']:
            if key in tensor_dict:
                tensor_dict[key] = tensor_dict[key].unsqueeze(0)
        
        self.profile_times['encode'].append(time.time() - start)
        return tensor_dict
    
    def _select_from_head(self, game, head_type, valid_options=None, possible_actions=None):
        """Run network forward pass and sample from specified head."""
        state_dict = self._encode_state(game)
        
        mask_start = time.time()
        mask = self._create_mask_for_head(
            head_type, valid_options, possible_actions, state_dict['action_masks']
        )
        self.profile_times['mask'].append(time.time() - mask_start)
        
        forward_start = time.time()
        with torch.no_grad():
            logits, value = self.network(state_dict, head_type=head_type, mask=None)
        self.profile_times['forward'].append(time.time() - forward_start)
        
        sample_start = time.time()
        logits = logits.squeeze(0)
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9
        
        if (mask == 1).sum() == 0:
            print(f"WARNING: No valid actions for {head_type}, enabling first option")
            mask[0] = 1.0
            masked_logits[0] = 0.0
        
        dist = torch.distributions.Categorical(logits=masked_logits.unsqueeze(0))
        action_idx = dist.sample().item()
        self.profile_times['sample'].append(time.time() - sample_start)
        
        return action_idx, masked_logits, value.item(), mask
    
    def print_profile_stats(self):
        print("\n=== RL Agent Profiling ===")
        for key, times in self.profile_times.items():
            if times:
                total = sum(times)
                avg = total / len(times)
                print(f"{key:10s}: {total:.3f}s total, {avg:.4f}s avg, {len(times)} calls")
        print("==========================\n")
    
    def _create_mask_for_head(self, head_type: str, valid_options, possible_actions, 
                            action_masks_dict: Dict) -> torch.Tensor:
        """Create validity mask for head output."""
        head_sizes = {
            'action': 8,
            'settlement': 54,
            'city': 54,
            'road': 72,
            'tile': 19,
            'steal_player': self.config.num_players,
            'dev_card': 5,
            'resource': 5,
            'resource2': 5,
            'trade_player': self.config.num_players,
            'trade_give_resource': 6,
            'trade_receive_resource': 6,
            'accept_trade': 2,
            'exchange_give': 5,
            'exchange_receive': 5,
            'discard': 5
        }
        
        mask = torch.zeros(head_sizes[head_type], device=self.device)
        
        if head_type == 'action':
            action_to_idx = {
                'build_settlement': 0,
                'build_city': 1,
                'build_road': 2,
                'buy_dev_card': 3,
                'play_dev_card': 4,
                'bank_trade': 5,
                'trade': 6,
                'end_turn': 7
            }
            
            if possible_actions:
                for action in possible_actions:
                    if action in action_to_idx:
                        idx = action_to_idx[action]
                        mask[idx] = 1.0
            
            if mask.sum() == 0:
                mask[7] = 1.0
        
        elif head_type in ['settlement', 'city', 'road', 'tile']:
            if valid_options is not None:
                if not isinstance(valid_options, torch.Tensor):
                    indices = torch.tensor(list(valid_options), dtype=torch.long, device=self.device)
                else:
                    indices = valid_options.to(self.device).long()
                
                mask[indices] = 1.0
            
            if mask.sum() == 0:
                mask[0] = 1.0
        
        elif head_type == 'dev_card':
            if 'card' in action_masks_dict:
                card_mask = action_masks_dict['card']
                if isinstance(card_mask, torch.Tensor):
                    mask[:len(card_mask)] = card_mask
                else:
                    mask[:len(card_mask)] = torch.tensor(card_mask, device=self.device, dtype=torch.float32)
            
            if mask.sum() == 0:
                mask[0] = 1.0
        
        elif head_type in ['steal_player', 'trade_player']:
            if valid_options is not None:
                if isinstance(valid_options, np.ndarray):
                    valid_options = valid_options.tolist()
                elif isinstance(valid_options, torch.Tensor):
                    valid_options = valid_options.cpu().tolist()
                elif isinstance(valid_options, set):
                    valid_options = list(valid_options)
                
                for player_id in valid_options:
                    if isinstance(player_id, (int, np.integer)) and 0 <= player_id < self.config.num_players:
                        mask[player_id] = 1.0
            else:
                mask[:] = 1.0
            
            # Exclude self
            if self.id < len(mask):
                mask[self.id] = 0.0
            
            if mask.sum() == 0:
                for i in range(self.config.num_players):
                    if i != self.id:
                        mask[i] = 1.0
                        break
        
        elif head_type in ['resource', 'resource2', 'discard', 'exchange_give', 'exchange_receive']:
            mask[:] = 1.0
        
        else:
            mask[:] = 1.0
        
        if mask.sum() == 0:
            print(f"ERROR: Created all-zero mask for {head_type}, enabling first option")
            mask[0] = 1.0
        
        return mask
    
    def choose_action(self, game, possible_actions: List[str]) -> str:
        self.action_stats['total_turns'] += 1
        
        action_idx, masked_logits, value, mask = self._select_from_head(
            game, 'action', possible_actions=possible_actions
        )
        
        action_map = {
            0: 'build_settlement', 1: 'build_city', 2: 'build_road',
            3: 'buy_dev_card', 4: 'play_dev_card', 5: 'bank_trade',
            6: 'trade', 7: 'end_turn',
        }
        
        chosen_action = action_map.get(action_idx, 'end_turn')
        
        if chosen_action not in possible_actions:
            print(f"⚠️ Action {chosen_action} (idx={action_idx}) not in {possible_actions}")
            print(f"   Mask: {mask.cpu().numpy()}")
            print(f"   Masked logits: {masked_logits.cpu().numpy()}")
            self.action_stats['invalid_attempts'] += 1
            chosen_action = 'end_turn' if 'end_turn' in possible_actions else possible_actions[0]
        
        self.action_stats['actions_chosen'][chosen_action] += 1
        
        if chosen_action == 'end_turn' and len(possible_actions) > 1:
            self.action_stats['early_turn_ends'] += 1
        
        return chosen_action
    
    def print_stats(self):
        print(f"\n=== RL Agent Stats ===")
        print(f"Total turns: {self.action_stats['total_turns']}")
        print(f"Invalid attempts: {self.action_stats['invalid_attempts']}")
        print(f"Early turn ends: {self.action_stats['early_turn_ends']}")
        print(f"Actions chosen: {dict(self.action_stats['actions_chosen'])}")
        print(f"End turn rate: {self.action_stats['actions_chosen']['end_turn'] / max(1, self.action_stats['total_turns']):.2%}")

    def choose_action(self, game, possible_actions: List[str]) -> str:
        action_idx, masked_logits, value, mask = self._select_from_head(
            game, 'action', possible_actions=possible_actions
        )
        
        action_map = {
            0: 'build_settlement', 1: 'build_city', 2: 'build_road',
            3: 'buy_dev_card', 4: 'play_dev_card', 5: 'bank_trade',
            6: 'trade', 7: 'end_turn',
        }
        
        chosen_action = action_map.get(action_idx, 'end_turn')
        
        if chosen_action not in possible_actions:
            print(f"⚠️ Masking bug: chose {chosen_action} not in {possible_actions}")
            return random.choice(possible_actions)
        
        return chosen_action
    
    def choose_settlement_location(self, game, valid_locations: List[int]) -> int:
        if not valid_locations:
            return 0
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'settlement', valid_options=valid_locations
        )
        
        return valid_locations[action_idx % len(valid_locations)]
    
    def choose_city_location(self, game, valid_locations: List[int]) -> int:
        if not valid_locations:
            return 0
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'city', valid_options=valid_locations
        )
        
        return valid_locations[action_idx % len(valid_locations)]
    
    def choose_road_location(self, game, valid_locations: List[int]) -> int:
        if not valid_locations:
            return 0
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'road', valid_options=valid_locations
        )
        
        return valid_locations[action_idx % len(valid_locations)]
    
    def choose_initial_settlement(self, game, valid_locations: List[int], round_num: int) -> int:
        return self.choose_settlement_location(game, valid_locations)
    
    def choose_initial_road(self, game, valid_edges: List[int], 
                          settlement_vertex: int, round_num: int) -> int:
        return self.choose_road_location(game, valid_edges)
    
    def choose_robber_placement(self, game, valid_hexes) -> int:
        valid_list = list(valid_hexes) if not isinstance(valid_hexes, list) else valid_hexes
        
        if not valid_list:
            return 0
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'tile', valid_options=valid_list
        )
        
        return valid_list[action_idx % len(valid_list)]
    
    def choose_steal_victim(self, game, possible_victims: List[int]) -> Optional[int]:
        if not possible_victims:
            return None
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'steal_player', valid_options=possible_victims
        )
        
        return possible_victims[action_idx % len(possible_victims)]
    
    def choose_dev_card_to_play(self, game, playable_cards: List[str]) -> Optional[str]:
        if not playable_cards:
            return None
        
        card_map = {
            'knight': 0, 'monopoly': 1, 'year_of_plenty': 2,
            'road_building': 3, 'victory_point': 4
        }
        
        valid_indices = [card_map[card] for card in playable_cards if card in card_map]
        
        if not valid_indices:
            return playable_cards[0]
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'dev_card', valid_options=valid_indices
        )
        
        reverse_map = {v: k for k, v in card_map.items()}
        chosen_card = reverse_map.get(action_idx)
        
        return chosen_card if chosen_card in playable_cards else playable_cards[0]
    
    def choose_monopoly_resource(self, game) -> str:
        resource_map = {0: 'wheat', 1: 'wood', 2: 'brick', 3: 'sheep', 4: 'ore'}
        action_idx, _, _, _ = self._select_from_head(game, 'resource')
        return resource_map.get(action_idx, 'wheat')
    
    def choose_year_of_plenty_resource(self, game) -> str:
        resource_map = {0: 'wheat', 1: 'wood', 2: 'brick', 3: 'sheep', 4: 'ore'}
        action_idx, _, _, _ = self._select_from_head(game, 'resource')
        return resource_map.get(action_idx, 'wheat')
    
    def choose_bank_trade(self, game, possible_trades: List[Dict]) -> Optional[Dict]:
        if not possible_trades:
            return None
        
        action_idx, _, _, _ = self._select_from_head(game, 'exchange_give')
        
        resource_map = {0: 'wheat', 1: 'wood', 2: 'brick', 3: 'sheep', 4: 'ore'}
        preferred_give = resource_map.get(action_idx, 'wheat')
        
        for trade in possible_trades:
            if trade['give'] == preferred_give:
                return trade
        
        return possible_trades[0]
    
    def choose_trade_strategy(self, game) -> str:
        """Map network output to strategy: 0=targeted, 1=broadcast, 2+=multiple."""
        action_idx, _, _, _ = self._select_from_head(game, 'trade_player')
        
        if action_idx == 0:
            return 'targeted'
        elif action_idx == 1:
            return 'broadcast'
        else:
            return 'multiple'
    
    def choose_trade_target(self, game) -> Optional[int]:
        others = [i for i in range(4) if i != self.id]
        
        if not others:
            return None
        
        action_idx, _, _, _ = self._select_from_head(
            game, 'trade_player', valid_options=others
        )
        
        if action_idx < len(others):
            return others[action_idx]
        else:
            return others[action_idx % len(others)]
        
    def choose_resources_to_discard(self, game, all_resources: List[str], 
                                   num_to_discard: int) -> List[str]:
        """Iteratively select resources to discard."""
        if num_to_discard >= len(all_resources):
            return all_resources
        
        discarded = []
        remaining = all_resources.copy()
        
        for _ in range(num_to_discard):
            if not remaining:
                break
            
            unique_resources = list(set(remaining))
            resource_map = {'wheat': 0, 'wood': 1, 'brick': 2, 'sheep': 3, 'ore': 4}
            valid_indices = [resource_map[r] for r in unique_resources if r in resource_map]
            
            if not valid_indices:
                chosen = random.choice(remaining)
            else:
                action_idx, _, _, _ = self._select_from_head(
                    game, 'discard', valid_options=valid_indices
                )
                
                reverse_map = {v: k for k, v in resource_map.items()}
                chosen = reverse_map.get(action_idx, unique_resources[0])
                
                if chosen not in remaining:
                    chosen = remaining[0]
            
            discarded.append(chosen)
            remaining.remove(chosen)
        
        return discarded
    
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

class PPOTrainer:
    def __init__(self, network, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 value_coef=0.5, entropy_coef=0.01, reward_config=None,
                 window_size=10):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        """from torch.optim.lr_scheduler import CosineAnnealingLR

        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_iterations,
            eta_min=1e-5
        )"""

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.opponent_network = copy.deepcopy(network)
        self.update_opponent_every = 10 # Update opponent to agent's network every N iterations
        
        #if entropy_coef < 0.01:
            #print(f"Warning: entropy_coef={entropy_coef} is too low, increasing to 0.05")
            #self.entropy_coef = 0.05  # Prevent deterministic policy
            
        self.max_grad_norm = 0.25
        
        self.reward_config = reward_config or RewardConfig()
        self.reward_calculator = RewardCalculator(self.reward_config)
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        
        self.window_size = window_size
        self.recent_wins = deque(maxlen=window_size)
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_lengths = deque(maxlen=window_size)

        self.iteration_metrics = {
            'iteration': [],
            'win_rate': [],
            'recent_win_rate': [],
            'mean_reward': [],
            'mean_length': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }

        self.reward_breakdowns = deque(maxlen=100)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def collect_episode(self, config, agent_class, other_player_class, player_id=0):
        self.network.eval()

        players = []
        for pid in range(config.num_players):
            if pid == player_id:
                player = agent_class(pid, player_colors[pid], config=config, 
                                    network=self.network)
            else:
                if other_player_class == CatanRLAgent:
                    # self-play: opponent uses lagged network
                    player = other_player_class(pid, player_colors[pid], config=config,
                                            network=self.opponent_network)
                else:
                    # heuristic opponent
                    player = other_player_class(pid, player_colors[pid])
            
            players.append(player)
        
        game = CatanGame(config=config, players=players)
        rl_player = game.players[player_id]

        experiences = {
            head_type: {
                'state_dicts': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'masks': [],
                'rewards': []
            }
            for head_type in self.network.heads.keys()
        }

        current_turn_decisions = []
        turn_reward_breakdowns = []
        
        game.setup_initial_placements()
        
        metrics_before = self.reward_calculator.extract_metrics(game, player_id)

        step = 0
        max_steps = 500
        
        episode_stats = {
            'actions_attempted': 0,
            'actions_succeeded': 0,
            'actions_by_type': defaultdict(int),
            'invalid_attempts': 0,
            'early_turn_ends': 0
        }
        
        with torch.no_grad():
            while not game.is_game_over() and step < max_steps:
                current_pid = game.current_player
                
                if current_pid == player_id:

                    current_turn_decisions.clear()
                    action_history = []
                    
                    roll = game.roll_dice(current_pid)

                    actions_taken = 0
                    max_actions = 5
                    trade_attempts_this_turn = 0
                    max_trade_attempts = 1
                    failed_actions = set()

                    while actions_taken < max_actions:
                        possible_actions = rl_player._get_possible_actions_strict(game)
                        possible_actions = [a for a in possible_actions if a not in failed_actions]
                        
                        if trade_attempts_this_turn >= max_trade_attempts:
                            possible_actions = [a for a in possible_actions if a not in ['trade', 'bank_trade']]

                        if not possible_actions:
                            break
                        
                        if actions_taken == 0 and len(possible_actions) == 1 and possible_actions[0] == 'end_turn':
                            break
                        
                        # encode 
                        state_dict = rl_player._encode_state(game)
                        
                        # mask
                        mask = rl_player._create_mask_for_head(
                            'action', None, possible_actions, state_dict['action_masks']
                        )
                        
                        # forward pass
                        logits, value = self.network(state_dict, head_type='action', mask=mask.unsqueeze(0))
                        
                        # sample action
                        dist = torch.distributions.Categorical(logits=logits)
                        action_idx = dist.sample()
                        log_prob = dist.log_prob(action_idx)
                        
                        # store decision
                        current_turn_decisions.append({
                            'head': 'action',
                            'state_dict': state_dict,
                            'action': action_idx.item(),
                            'log_prob': log_prob.item(),
                            'value': value.item(),
                            'mask': mask
                        })
                        
                        # map action
                        action_map = {
                            0: 'build_settlement',
                            1: 'build_city',
                            2: 'build_road',
                            3: 'buy_dev_card',
                            4: 'play_dev_card',
                            5: 'bank_trade',
                            6: 'trade',
                            7: 'end_turn'
                        }
                        chosen_action = action_map.get(action_idx.item(), 'end_turn')
                        
                        if chosen_action in ['trade', 'bank_trade']:
                            trade_attempts_this_turn += 1
                            
                        if chosen_action == 'end_turn':
                            break
                        
                        action_success = self._execute_action_with_tracking(
                            game, player_id, chosen_action, current_turn_decisions, action_history
                        )
                        
                        episode_stats['actions_attempted'] += 1
                        if action_success:
                            episode_stats['actions_succeeded'] += 1
                            episode_stats['actions_by_type'][chosen_action] += 1
                            actions_taken += 1
                            failed_actions.clear()
                        else:
                            if chosen_action not in ['trade']:
                                episode_stats['invalid_attempts'] += 1
                            failed_actions.add(chosen_action)
                            action_history.append('invalid_action')
                            if len(failed_actions) >= len(possible_actions) - 1:
                                break

                    if actions_taken < 2 and actions_taken > 0:
                        episode_stats['early_turn_ends'] += 1

                    # compute rewards
                    metrics_after = self.reward_calculator.extract_metrics(game, player_id)
                    turn_reward, breakdown = self.reward_calculator.compute_turn_reward(
                        metrics_before, metrics_after, game, player_id, 
                        action_history, game.turn
                    )
                    #print(f"Reward breakdown: {breakdown}")
                    turn_reward_breakdowns.append(breakdown)
                    metrics_before = metrics_after
                    
                    for decision in current_turn_decisions:
                        head = decision['head']
                        experiences[head]['state_dicts'].append(decision['state_dict'])
                        experiences[head]['actions'].append(decision['action'])
                        experiences[head]['log_probs'].append(decision['log_prob'])
                        experiences[head]['values'].append(decision['value'])
                        experiences[head]['masks'].append(decision['mask'])
                        experiences[head]['rewards'].append(turn_reward)
                
                else:
                    roll = game.roll_dice(current_pid)
                    game.players[current_pid].turn(game)
                
                game.current_player = (game.current_player + 1) % config.num_players
                if game.current_player == 0:
                    game.turn += 1
                
                step += 1
        
        win = 0
        if game.is_game_over():
            scores = [(pid, game.get_player_score(pid)) for pid in range(config.num_players)]
            max_score = max(score for _, score in scores)
            winners = [pid for pid, score in scores if score == max_score]
            
            if len(winners) == 1 and winners[0] == player_id:
                win = 1
            elif len(winners) > 1 and player_id in winners:
                win = 0 # ties dont count, leads to reward hacking where gaents run out time in self-play and get consistent rewards
            else:
                win = 0
            
            #player_score = game.get_player_score(player_id)
            #opponent_scores = [game.get_player_score(pid) for pid in range(config.num_players) if pid != player_id]
            
            terminal_reward = self.reward_calculator.compute_terminal_reward(game, player_id)
            
            for head_type in experiences:
                if experiences[head_type]['rewards']:
                    experiences[head_type]['rewards'][-1] += terminal_reward
        
        total_reward = sum(experiences['action']['rewards']) if experiences['action']['rewards'] else 0
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(game.turn)
        self.win_rates.append(win)
        self.reward_breakdowns.append(turn_reward_breakdowns)
        
        self.recent_wins.append(win)
        self.recent_rewards.append(total_reward)
        self.recent_lengths.append(game.turn)
        
        for head_type in experiences:
            exp = experiences[head_type]
            if exp['state_dicts']:
                exp['actions'] = np.array(exp['actions'], dtype=np.int64)
                exp['log_probs'] = np.array(exp['log_probs'], dtype=np.float32)
                exp['values'] = np.array(exp['values'], dtype=np.float32)
                exp['rewards'] = np.array(exp['rewards'], dtype=np.float32)
                exp['masks'] = torch.stack(exp['masks'])
            else:
                exp['state_dicts'] = []
                exp['actions'] = np.array([])
                exp['log_probs'] = np.array([])
                exp['values'] = np.array([])
                exp['rewards'] = np.array([])
                exp['masks'] = None
        
        return experiences
    
    def _execute_action_with_tracking(self, game, player_id, action, decision_list, action_history):
        """Execute action with tracking"""
        player = game.players[player_id]
        
        if action == 'build_settlement':
            if not game.can_afford(player_id, 'settlement'):
                return False
                
            valid = game.get_valid_settlement_locations(player_id)
            if not valid:
                return False
            
            state_dict = player._encode_state(game)
            action_idx, logits, value, mask = player._select_from_head(
                game, 'settlement', valid_options=valid
            )
            
            location = valid[action_idx % len(valid)]
            
            success = game.build_settlement(player_id, location)
            if success:
                action_history.append('build_settlement')
            return success
        
        elif action == 'build_city':
            if not game.can_afford(player_id, 'city'):
                return False
                
            valid = game.get_valid_city_locations(player_id)
            if not valid:
                return False
            
            action_idx, _, _, _= player._select_from_head(
                game, 'city', valid_options=valid
            )
            
            location = valid[action_idx % len(valid)]
            
            success = game.build_city(player_id, location)
            if success:
                action_history.append('build_city')
            return success
        
        elif action == 'build_road':
            if not game.can_afford(player_id, 'road'):
                return False
                
            valid = game.get_valid_road_locations(player_id)
            if not valid:
                return False
            
            action_idx, _, _, _ = player._select_from_head(
                game, 'road', valid_options=valid
            )
            
            location = valid[action_idx % len(valid)]
            
            success = game.build_road(player_id, location)
            if success:
                action_history.append('build_road')
            return success
        
        elif action == 'buy_dev_card':
            if not game.can_afford(player_id, 'development_card'):
                return False
            if len(game.dev_card_deck) == 0:
                return False
            
            success = game.buy_development_card(player_id)
            if success:
                action_history.append('buy_dev_card')
            return success
        
        elif action == 'bank_trade':
            inv = game.get_player_inventory(player_id)
            ratios = game.get_player_trade_ratios(player_id)
            can_trade = any(inv.get(res, 0) >= ratio for res, ratio in ratios.items())
            
            if not can_trade:
                return False
            
            success = player._try_bank_trade(game)
            if success:
                trade_ratios = game.get_player_trade_ratios(player_id)
                has_good_port = any(ratio < 4 for ratio in trade_ratios.values())
                if has_good_port:
                    action_history.append('port_trade')
                else:
                    action_history.append('bank_trade_4to1')
            return success
        
        elif action == 'trade':
            inv = game.get_player_inventory(player_id)
            if sum(inv.values()) == 0:
                return False
            
            success = player._try_trade(game)
            if success:
                action_history.append('trade_success')
            return success
        
        elif action == 'play_dev_card':
            playable = player._get_playable_dev_cards(game)
            if not playable:
                return False
            
            card = player.choose_dev_card_to_play(game, playable)
            if card:
                success = game.play_development_card(player_id, card)
                if success:
                    action_history.append(f'play_{card}')
                return success
            return False
        
        return False
    
    def _batch_state_dicts(self, state_dict_list):
        if not state_dict_list:
            return None
        
        #batch_size = len(state_dict_list)
        batched = {}
        
        tensor_fields = ['hex_features', 'vertex_features', 'edge_features', 
                        'player_vector', 'global_context']
        
        for field in tensor_fields:
            tensors = [sd[field] for sd in state_dict_list]
            batched[field] = torch.cat(tensors, dim=0)
        
        batched['graph_edges'] = state_dict_list[0]['graph_edges']
        batched['edge_types'] = state_dict_list[0]['edge_types']
        batched['action_masks'] = [sd['action_masks'] for sd in state_dict_list]
        
        return batched
    
    def update_policy(self, episodes, num_update_epochs=2, batch_size=256):
        """Update policy with proper action masking"""
        self.network.train()

        for param in self.network.parameters():
            param.requires_grad = True
        
        total_losses = {'policy': [], 'value': [], 'entropy': []}
        
        for head_type in self.network.heads.keys():
            all_state_dicts = []
            all_actions = []
            all_old_log_probs = []
            all_returns = []
            all_advantages = []
            all_masks = []
            
            for episode in episodes:
                exp = episode[head_type]
                if len(exp['state_dicts']) == 0:
                    continue
                
                returns, advantages = self.compute_gae(exp['rewards'], exp['values'])
                
                all_state_dicts.extend(exp['state_dicts'])
                all_actions.append(exp['actions'])
                all_old_log_probs.append(exp['log_probs'])
                all_returns.append(returns)
                all_advantages.append(advantages)
                if exp['masks'] is not None:
                    all_masks.append(exp['masks'])
            
            if not all_state_dicts:
                continue
            
            # normalize
            returns_np = np.concatenate(all_returns)
            returns_mean = returns_np.mean()
            returns_std = returns_np.std() + 1e-8
            returns_normalized = (returns_np - returns_mean) / returns_std
            
            actions = torch.tensor(np.concatenate(all_actions), device=self.device, dtype=torch.long)
            old_log_probs = torch.tensor(np.concatenate(all_old_log_probs), device=self.device, dtype=torch.float32)
            returns = torch.tensor(returns_normalized, device=self.device, dtype=torch.float32)
            advantages = torch.tensor(np.concatenate(all_advantages), device=self.device, dtype=torch.float32)
            masks = torch.cat(all_masks, dim=0) if all_masks else None

            if torch.isnan(returns).any() or torch.isinf(returns).any():
                print(f"WARNING: NaN/Inf in returns for {head_type}, skipping")
                continue
            
            num_samples = len(all_state_dicts)
            
            for epoch in range(num_update_epochs):
                indices = torch.randperm(num_samples, device=self.device)
                
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    idx = indices[start:end]
                    
                    batch_state_dicts = [all_state_dicts[i.item()] for i in idx]
                    batched_state = self._batch_state_dicts(batch_state_dicts)
                    
                    batch_actions = actions[idx]
                    batch_old_log_probs = old_log_probs[idx]
                    batch_returns = returns[idx]
                    batch_advantages = advantages[idx]
                    batch_masks = masks[idx] if masks is not None else None
                    
                    logits, values = self.network(batched_state, head_type=head_type, mask=None)
                    
                    if torch.isnan(logits).any() or torch.isnan(values).any():
                        print(f"CRITICAL: NaN in {head_type} output")
                        return {
                            'policy_loss': float('nan'),
                            'value_loss': float('nan'),
                            'entropy': float('nan')
                        }
                    
                    if batch_masks is not None:
                        masked_logits = logits.clone()
                        # set invalid actions to -inf (will have prob=0 after softmax)
                        masked_logits[batch_masks == 0] = float('-inf')
                        
                        valid_actions_per_sample = (batch_masks == 1).sum(dim=1)
                        if (valid_actions_per_sample == 0).any():
                            print(f"WARNING: Some samples in {head_type} have no valid actions!")
                            all_masked_samples = (valid_actions_per_sample == 0)
                            masked_logits[all_masked_samples, 0] = 0.0
                    else:
                        masked_logits = logits
                    
                    # get distribution from masked logits
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    
                    new_log_probs = dist.log_prob(batch_actions)
                    
                    entropy = dist.entropy().mean()
                    
                    # clipping PPO
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # MSE for value
                    value_pred = values.squeeze(-1) if values.dim() > 1 else values
                    value_loss = 0.5 * torch.nn.functional.mse_loss(value_pred, batch_returns)
                    
                    # full loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    if torch.isnan(loss):
                        print(f"WARNING: NaN loss for {head_type}, skipping batch")
                        print(f"  Policy loss: {policy_loss.item()}, Value loss: {value_loss.item()}, Entropy: {entropy.item()}")
                        print(f"  Ratio range: [{ratio.min().item():.3f}, {ratio.max().item():.3f}]")
                        continue

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    total_losses['policy'].append(policy_loss.item())
                    total_losses['value'].append(value_loss.item())
                    total_losses['entropy'].append(entropy.item())
        
        return {
            'policy_loss': np.mean(total_losses['policy']) if total_losses['policy'] else 0,
            'value_loss': np.mean(total_losses['value']) if total_losses['value'] else 0,
            'entropy': np.mean(total_losses['entropy']) if total_losses['entropy'] else 0
        }
    
    def compute_gae(self, rewards, values, gamma=None, lam=0.95):
        if gamma is None:
            gamma = self.gamma
        
        advantages = []
        gae = 0
        
        # advantage calculation
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        # returns: R_t = A_t + V(s_t)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train(self, config, agent_class, other_player_class, num_iterations=100, episodes_per_iter=4, num_update_epochs=4, batch_size=64, save_dir = 'RLmodels/ppo_models/'):
        # reset iteration metrics for each run (creates wack plots if not)
        self.iteration_metrics = {
            'iteration': [],
            'win_rate': [],
            'recent_win_rate': [],
            'mean_reward': [],
            'mean_length': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        
        }
        game_type = (f"{config.num_players}p_"#vs{other_player_class.__name__[0]}_"
                    f"{config.max_turns}maxt_{int(config.allow_trading)}t_"
                    f"{int(config.allow_bank_trading)}bt")
        os.makedirs(save_dir, exist_ok=True)

        plural = 's'
        print(f"Training one {agent_class.__name__} agent against {int_to_str_dict[config.num_players-1]} "
              f"{other_player_class.__name__}{plural if config.num_players-1 > 1 else ''} for {bold_print(num_iterations)} iterations w/ {bold_print(episodes_per_iter)} episodes each on {self.device}.\n")
        print(f"Save directory: {save_dir}")
        print(f"Rolling window size: {self.window_size} episodes")
        print("-" * 80)
        
        for iteration in range(1, num_iterations + 1):
            episodes = []
            
            with tqdm(range(episodes_per_iter),
                desc=f"Iter {iteration:3d}/{num_iterations}",
                leave=False,
                ncols=100,
                file=sys.stdout) as pbar:
                
                for ep in pbar:
                    try:
                        episode_data = self.collect_episode(config, agent_class, other_player_class)
                        episodes.append(episode_data)
                    except Exception as e:
                        pbar.write(f"Error in episode {ep}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            if episodes:
                metrics = self.update_policy(episodes, num_update_epochs=num_update_epochs, batch_size=batch_size)
                #self.scheduler.step()
            else:
                metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            cumulative_win_rate = np.mean(self.win_rates) if self.win_rates else 0
            
            # compute rolling window win rate
            recent_win_rate = np.mean(self.recent_wins) if self.recent_wins else 0

            # store iteration metrics for plotting
            self.iteration_metrics['iteration'].append(iteration)
            self.iteration_metrics['win_rate'].append(cumulative_win_rate)
            self.iteration_metrics['recent_win_rate'].append(recent_win_rate)
            self.iteration_metrics['mean_reward'].append(avg_reward)
            self.iteration_metrics['mean_length'].append(avg_length)
            self.iteration_metrics['policy_loss'].append(metrics['policy_loss'])
            self.iteration_metrics['value_loss'].append(metrics['value_loss'])
            self.iteration_metrics['entropy'].append(metrics['entropy'])

            cumulative_wr_display = np.round(cumulative_win_rate * 100, 2)
            recent_wr_display = np.round(recent_win_rate * 100, 2)
            
            colored_recent = color_print(recent_wr_display, Range=(0, 90))

            print(
                f"Iter {iteration:3d}/{num_iterations} | "
                f"Win rate: {cumulative_wr_display:5.1f}% (recent: {colored_recent}%) | "
                f"Reward: {avg_reward:7.2f} | "
                f"Length: {avg_length:5.1f} | "
                f"Policy: {metrics['policy_loss']:6.3f} | "
                f"Value: {metrics['value_loss']:6.3f} | "
                f"Entropy: {metrics['entropy']:6.3f}",
                flush=True
            )
            
            if iteration % self.update_opponent_every == 0:
                self.opponent_network.load_state_dict(self.network.state_dict())

            if iteration % 10 == 0:
                torch.save({
                    'iteration': iteration,
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward_config': self.reward_config,
                    'metrics': {
                        'rewards': list(self.episode_rewards),
                        'lengths': list(self.episode_lengths),
                        'win_rates': list(self.win_rates),
                        'recent_wins': list(self.recent_wins),
                        'iteration_metrics': self.iteration_metrics
                    }
                }, f'{save_dir}/PPO_{iteration}.pt')
                
                print(f" ✓ Checkpoint saved to {save_dir}/PPO_{iteration}.pt")
                self.plot_training_progress(save_dir, iteration)

        print("\n" + "="*80)
        print("Training complete!")
        print(f"Cumulative win rate: {cumulative_win_rate:.1%}")
        print(f"Recent win rate (last {self.window_size}): {recent_win_rate:.1%}")
        print(f"Final avg reward: {avg_reward:.2f}")
        
        self.plot_training_progress(save_dir, num_iterations, final=True)
        
        return self.network
    
    def plot_training_progress(self, save_dir, iteration, final=False):
        """Plot training progress for current stage"""
        if len(self.iteration_metrics['iteration']) < 2:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - Iteration {iteration}', fontsize=16, fontweight='bold')
        
        iters = self.iteration_metrics['iteration']
        
        # Cumulative vs. rolling win rates
        ax = axes[0, 0]
        ax.plot(iters, self.iteration_metrics['win_rate'], 
                label='Cumulative WR', alpha=0.5, marker='o', markersize=3, color='blue')
        ax.plot(iters, self.iteration_metrics['recent_win_rate'],
                label=f'Rolling WR (last {self.window_size})', linewidth=2, color='green')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # rewards
        ax = axes[0, 1]
        ax.plot(iters, self.iteration_metrics['mean_reward'], marker='o', markersize=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Mean Episode Reward')
        ax.grid(True, alpha=0.3)
        
        # game length
        ax = axes[0, 2]
        ax.plot(iters, self.iteration_metrics['mean_length'], 
                marker='o', markersize=3, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Length')
        ax.set_title('Mean Episode Length')
        ax.grid(True, alpha=0.3)
        
        # policy loss
        ax = axes[1, 0]
        ax.plot(iters, self.iteration_metrics['policy_loss'], 
                marker='o', markersize=3, color='orange')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # value loss
        ax = axes[1, 1]
        ax.plot(iters, self.iteration_metrics['value_loss'],
                marker='o', markersize=3, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss (log scale)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # entropy
        ax = axes[1, 2]
        ax.plot(iters, self.iteration_metrics['entropy'],
                marker='o', markersize=3, color='purple')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
        
        # win rate comparison
        ax = axes[2, 0]
        width = 0.35
        x = np.arange(len(iters[-10:]))
        if len(iters) >= 10:
            cumulative = self.iteration_metrics['win_rate'][-10:]
            recent = self.iteration_metrics['recent_win_rate'][-10:]
            ax.bar(x - width/2, cumulative, width, label='Cumulative', alpha=0.7)
            ax.bar(x + width/2, recent, width, label='Recent', alpha=0.7)
            ax.set_xlabel('Last 10 Iterations')
            ax.set_ylabel('Win Rate')
            ax.set_title('Cumulative vs Recent Win Rate')
            ax.set_xticks(x)
            ax.set_xticklabels(iters[-10:])
            ax.legend()
            ax.set_ylim([0, 1])
        
        # win rate
        ax = axes[2, 1]
        if len(self.iteration_metrics['recent_win_rate']) > 5:
            window = 5
            smoothed = np.convolve(self.iteration_metrics['recent_win_rate'], 
                                np.ones(window)/window, mode='valid')
            ax.plot(iters[window-1:], smoothed, linewidth=2, label=f'{window}-iter MA')
            ax.plot(iters, self.iteration_metrics['recent_win_rate'], 
                alpha=0.3, label='Raw', marker='.')
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Smoothed Recent Win Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        ax = axes[2, 2]
        ax.axis('off')

        if len(self.recent_wins) > 0:
            recent_wr = np.mean(self.recent_wins) * 100
            cumulative_wr = np.mean(self.win_rates) * 100
            improvement = recent_wr - cumulative_wr
            
            summary_text = f"""
            Training Summary (Iter {iteration})
            
            Win Rates:
            • Recent ({self.window_size} eps): {recent_wr:.1f}%
            • Cumulative (all): {cumulative_wr:.1f}%
            • Improvement: {improvement:+.1f}%
            
            Performance:
            • Avg Reward: {self.iteration_metrics['mean_reward'][-1]:.1f}
            • Avg Length: {self.iteration_metrics['mean_length'][-1]:.1f}
            
            Learning:
            • Policy Loss: {self.iteration_metrics['policy_loss'][-1]:.3f}
            • Value Loss: {self.iteration_metrics['value_loss'][-1]:.3f}
            • Entropy: {self.iteration_metrics['entropy'][-1]:.3f}
            """
            ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        suffix = '_final' if final else ''
        plot_path = f'{save_dir}/training_progress{suffix}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if not final:
            print(f" ✓ Plot saved to {plot_path}")
        else:
            print(f"\n✓ Final plot saved to {plot_path}")
    
class SupervisedDataCollector:
    def __init__(self, encoder, checkpoint_pool_size=10):
        self.encoder = encoder
        self.demonstrations = []
        self.checkpoint_pool = []
        self.checkpoint_pool_size = checkpoint_pool_size
    
    def add_checkpoint(self, network_state_dict, config):
        """Add trained network to opponent pool for diversity."""
        self.checkpoint_pool.append({
            'state_dict': copy.deepcopy(network_state_dict),
            'config': config
        })
        if len(self.checkpoint_pool) > self.checkpoint_pool_size:
            self.checkpoint_pool.pop(0)
        print(f"  ✓ Checkpoint added to pool (size: {len(self.checkpoint_pool)})")
    
    def collect_diverse_games(self, config, num_games, current_network=None, 
                             opponent_distribution=None):
        """Collect games against varied opponents"""
        if opponent_distribution is None:
            if len(self.checkpoint_pool) > 0:
                opponent_distribution = {
                    'random': 0.25,
                    'greedy': 0.50,
                    'checkpoint': 0.25
                }
            else:
                opponent_distribution = {
                    'random': 0.33,
                    'greedy': 0.67
                }
        
        games_by_opponent = defaultdict(int)
        
        for game_num in tqdm(range(num_games), desc="Collecting diverse games"):
            opponent_types = list(opponent_distribution.keys())
            weights = list(opponent_distribution.values())
            opp_type = random.choices(opponent_types, weights=weights)[0]
            
            if opp_type == 'checkpoint' and len(self.checkpoint_pool) == 0:
                opp_type = random.choice(['random', 'greedy'])
            
            games_by_opponent[opp_type] += 1
            
            episode_data = self.collect_from_game(None, player_id=0)
            expert = AgentLogging(0, 'red', self, episode_data)
            
            opponents = []
            for i in range(1, config.num_players):
                if opp_type == 'random':
                    opp = RandomAgent(i, player_colors[i])
                elif opp_type == 'greedy':
                    opp = GreedyAgent(i, player_colors[i])
                else:
                    checkpoint = random.choice(self.checkpoint_pool)
                    opp_network = CatanGNNTransformerNetwork(
                        config,
                        hidden_dim=64,
                        num_gnn_layers=1,
                        num_transformer_layers=1,
                        num_attention_heads=2,
                        dropout=0.1
                    )
                    opp_network.load_state_dict(checkpoint['state_dict'])
                    opp = CatanRLAgent(i, player_colors[i], config=config, 
                                      network=opp_network, training_mode=False)
                
                opponents.append(opp)
            
            players = [expert] + opponents
            game = CatanGame(config=config, players=players)
            game.setup_initial_placements()
            
            while not game.is_game_over():
                roll = game.roll_dice(game.current_player)
                game.players[game.current_player].turn(game)
                game.current_player = (game.current_player + 1) % config.num_players
                if game.current_player == 0:
                    game.turn += 1
            
            self.finalize_episode(episode_data, game, player_id=0)
        
        print(f"\n📊 Collection summary:")
        for opp_type, count in games_by_opponent.items():
            print(f"  {opp_type}: {count} games ({count/num_games*100:.1f}%)")

    def record_decision(self, episode_data, game, player_id, head_type, 
                   action_idx, possible_actions=None, valid_options=None):
        """Record decision with float16 compression."""
        state = self.encoder.encode(game, player_id)
        
        state_compressed = {
            'hex_features': state['hex_features'].astype(np.float16),
            'vertex_features': state['vertex_features'].astype(np.float16),
            'edge_features': state['edge_features'].astype(np.float16),
            'player_vector': state['player_vector'].astype(np.float16),
            'global_context': state['global_context'].astype(np.float16),
            'action_masks': state.get('action_masks'),
        }
        
        if head_type == 'action':
            mask = self._create_action_mask(possible_actions)
        elif head_type in ['settlement', 'city']:
            mask = self._create_location_mask(valid_options, max_size=54)
        elif head_type == 'road':
            mask = self._create_location_mask(valid_options, max_size=72)
        elif head_type == 'tile':
            mask = self._create_location_mask(valid_options, max_size=19)
        else:
            mask = None
        
        episode_data['states'].append(state_compressed)
        episode_data['actions'].append(np.int16(action_idx))
        episode_data['head_types'].append(head_type)
        episode_data['masks'].append(mask)
    
    def _create_action_mask(self, possible_actions):
        mask = np.zeros(8, dtype=np.float32)
        action_map = {
            'build_settlement': 0, 'build_city': 1, 'build_road': 2,
            'buy_dev_card': 3, 'play_dev_card': 4, 'bank_trade': 5,
            'trade': 6, 'end_turn': 7
        }
        for action in possible_actions:
            idx = action_map.get(action, 7)
            mask[idx] = 1.0
        return mask
    
    def _create_location_mask(self, valid_options, max_size):
        mask = np.zeros(max_size, dtype=np.float32)
        if valid_options:
            for loc in valid_options:
                if 0 <= loc < max_size:
                    mask[loc] = 1.0
        return mask
    
    def collect_from_game(self, game, player_id):
        return {
            'states': [],
            'actions': [],
            'head_types': [],
            'masks': [],
            'player_id': player_id
        }
    
    def finalize_episode(self, episode_data, game, player_id):
        if episode_data['states']:
            scores = [(pid, game.get_player_score(pid)) for pid in range(game.config.num_players)]
            winner = max(scores, key=lambda x: x[1])[0]
            episode_data['metadata'] = {
                'won': winner == player_id,
                'final_score': game.get_player_score(player_id),
                'winner_id': winner,
                'turns': game.turn,
                'game_ended': game.is_game_over()
            }
            self.demonstrations.append(episode_data)
        else:
            print(f"  Episode empty.")
    
    def save(self, filepath):
        """Save with gzip compression."""
        filepath_gz = filepath.replace('.pt', '.pkl.gz')
        
        data = {
            'demonstrations': self.demonstrations,
            'checkpoint_pool': self.checkpoint_pool
        }
        
        os.makedirs(os.path.dirname(filepath_gz), exist_ok=True)
        with gzip.open(filepath_gz, 'wb', compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = os.path.getsize(filepath_gz) / (1024 * 1024)
        print(f"✓ Saved {len(self.demonstrations)} games to {filepath_gz}")
        print(f"  File size: {size_mb:.1f} MB ({size_mb/len(self.demonstrations):.2f} MB/game)")
    
    def load(self, filepath):
        """Load with automatic format detection."""
        filepath_gz = filepath.replace('.pt', '.pkl.gz')
        
        if os.path.exists(filepath_gz):
            print(f"Loading compressed format...")
            with gzip.open(filepath_gz, 'rb') as f:
                data = pickle.load(f)
        elif os.path.exists(filepath):
            print(f"Loading uncompressed format...")
            data = torch.load(filepath)
        else:
            raise FileNotFoundError(f"Could not find {filepath_gz} or {filepath}")
        
        self.demonstrations = data['demonstrations']
        if 'checkpoint_pool' in data:
            self.checkpoint_pool = data['checkpoint_pool']
        
        print(f"✓ Loaded {len(self.demonstrations)} demonstrations")
    
    def plot_metrics(self):
        if self.demonstrations is None:
            print("No demonstrations loaded!")
            return
        demonstrations = self.demonstrations
        
        if not demonstrations:
            print("No demonstrations to plot!")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Action type distribution
        ax1 = fig.add_subplot(gs[0, :2])
        
        head_type_counts = Counter()
        for episode in demonstrations:
            head_type_counts.update(episode['head_types'])
        
        head_types = sorted(head_type_counts.keys())
        counts = [head_type_counts[ht] for ht in head_types]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(head_types)))
        bars = ax1.bar(head_types, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Decision Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Decision Types', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Settlement location heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        
        settlement_locations = []
        for episode in demonstrations:
            for action, head_type in zip(episode['actions'], episode['head_types']):
                if head_type == 'settlement':
                    settlement_locations.append(action)
        
        if settlement_locations:
            location_counts = Counter(settlement_locations)
            
            heatmap_data = np.zeros((9, 6))
            for vertex_id, count in location_counts.items():
                row = vertex_id // 6
                col = vertex_id % 6
                if row < 9 and col < 6:
                    heatmap_data[row, col] = count
            
            im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax2.set_title('Settlement Location Frequency\n(Vertex ID Heatmap)', 
                        fontsize=14, fontweight='bold')
            ax2.set_xlabel('Column', fontsize=11)
            ax2.set_ylabel('Row', fontsize=11)
            
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Frequency', fontsize=10)
            
            for i in range(9):
                for j in range(6):
                    if heatmap_data[i, j] > 0:
                        text = ax2.text(j, i, f'{int(heatmap_data[i, j])}',
                                    ha="center", va="center", color="black", fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No settlement data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Settlement Location Frequency', fontsize=14, fontweight='bold')
        
        # Action choices distribution
        ax3 = fig.add_subplot(gs[1, :2])
        
        action_choices = []
        for episode in demonstrations:
            for action, head_type in zip(episode['actions'], episode['head_types']):
                if head_type == 'action':
                    action_choices.append(action)
        
        if action_choices:
            action_map = {
                0: 'build_settlement', 1: 'build_city', 2: 'build_road',
                3: 'buy_dev_card', 4: 'play_dev_card', 5: 'bank_trade',
                6: 'trade', 7: 'end_turn',
            }
            action_counts = Counter(action_choices)
            actions = [action_map.get(a, f'Action {a}') for a in sorted(action_counts.keys())]
            counts = [action_counts[a] for a in sorted(action_counts.keys())]
            
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(actions)))
            bars = ax3.bar(actions, counts, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax3.set_xlabel('Action Type', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax3.set_title('High-Level Action Distribution', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(axis='y', alpha=0.3)
        
        # Resource choices
        ax4 = fig.add_subplot(gs[1, 2:])
        
        resource_choices = []
        for episode in demonstrations:
            for action, head_type in zip(episode['actions'], episode['head_types']):
                if head_type == 'resource':
                    resource_choices.append(action)
        
        if resource_choices:
            resource_map = {0: 'Wheat', 1: 'Wood', 2: 'Brick', 3: 'Sheep', 4: 'Ore'}
            resource_counts = Counter(resource_choices)
            
            resources = [resource_map.get(r, f'Resource {r}') for r in sorted(resource_counts.keys())]
            counts = [resource_counts[r] for r in sorted(resource_counts.keys())]
            
            colors = ['#FFD700', '#8B4513', '#DC143C', '#90EE90', '#808080'][:len(resources)]
            bars = ax4.bar(resources, counts, color=colors, edgecolor='black', linewidth=1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax4.set_xlabel('Resource Type', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Times Selected', fontsize=12, fontweight='bold')
            ax4.set_title('Resource Selection (Dev Cards)', fontsize=14, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No resource selection data', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Resource Selection (Dev Cards)', fontsize=14, fontweight='bold')
        
        # Episode metrics
        ax5 = fig.add_subplot(gs[2, 0])
        
        episode_lengths = [len(ep['actions']) for ep in demonstrations]
        
        ax5.hist(episode_lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
        ax5.set_xlabel('Decisions per Episode', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Episode Length Distribution', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # Win rate & scores
        ax6 = fig.add_subplot(gs[2, 1])
        
        wins = sum(1 for ep in demonstrations 
                if ep.get('metadata', {}).get('won', False))
        win_rate = wins / len(demonstrations) if demonstrations else 0
        
        ax6.bar(['Wins', 'Losses'], [wins, len(demonstrations) - wins],
            color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
        ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax6.set_title(f'Win Rate: {win_rate:.1%}', fontsize=13, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate([wins, len(demonstrations) - wins]):
            ax6.text(i, v, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Score distribution
        ax7 = fig.add_subplot(gs[2, 2])
        
        final_scores = [ep.get('metadata', {}).get('final_score') 
                        for ep in demonstrations]
        final_scores = [s for s in final_scores if s is not None]
        
        if final_scores:
            ax7.hist(final_scores, bins=15, color='coral', edgecolor='black', alpha=0.7)
            ax7.axvline(np.mean(final_scores), color='blue', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(final_scores):.1f}')
            ax7.set_xlabel('Final Score', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax7.set_title('Score Distribution', fontsize=13, fontweight='bold')
            ax7.legend()
            ax7.grid(axis='y', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No score data available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Score Distribution', fontsize=13, fontweight='bold')
        
        # Data quality metrics
        ax8 = fig.add_subplot(gs[2, 3])
        
        total_decisions = sum(len(ep['actions']) for ep in demonstrations)
        unique_settlement_locs = len(set(
            action for ep in demonstrations 
            for action, ht in zip(ep['actions'], ep['head_types']) 
            if ht == 'settlement'
        ))
        unique_road_locs = len(set(
            action for ep in demonstrations 
            for action, ht in zip(ep['actions'], ep['head_types']) 
            if ht == 'road'
        ))
        
        metrics = {
            'Episodes': len(demonstrations),
            'Total Decisions': total_decisions,
            'Avg Decisions/Ep': total_decisions / len(demonstrations) if demonstrations else 0,
            'Unique Settlements': unique_settlement_locs,
            'Unique Roads': unique_road_locs
        }
        
        ax8.axis('off')
        
        table_data = [[k, f'{v:.1f}' if isinstance(v, float) else str(v)] 
                    for k, v in metrics.items()]
        
        table = ax8.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(metrics) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                table[(i, 1)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax8.set_title('Dataset Summary', fontsize=13, fontweight='bold', pad=20)
        
        fig.suptitle('Supervised Data Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(demonstrations)}")
        print(f"Total Decisions: {total_decisions}")
        print(f"Average Decisions per Episode: {total_decisions / len(demonstrations):.1f}")
        print(f"\nWin Rate: {win_rate:.1%}")
        if final_scores:
            print(f"Average Score: {np.mean(final_scores):.2f}")
        else:
            print("Average Score: N/A")
        print(f"\nDecision Type Breakdown:")
        for head_type in sorted(head_type_counts.keys()):
            pct = 100 * head_type_counts[head_type] / total_decisions
            print(f"  {head_type:20s}: {head_type_counts[head_type]:4d} ({pct:5.1f}%)")
        print(f"\nDiversity Metrics:")
        print(f"  Unique Settlement Locations: {unique_settlement_locs}/54")
        print(f"  Unique Road Locations: {unique_road_locs}/72")
        print("="*60)
        
        plt.show()
        
        return fig
    
class AgentLogging(GreedyAgent):
    """Wrap expert agent to log ALL decisions"""
    
    def __init__(self, id, color, collector, episode_data):
        super().__init__(id, color)
        self.collector = collector
        self.episode_data = episode_data
    
    def choose_action(self, game, possible_actions):
        action = super().choose_action(game, possible_actions)
        
        if action is not None:
            action_map = {
                'build_settlement': 0, 'build_city': 1, 'build_road': 2,
                'buy_dev_card': 3, 'play_dev_card': 4, 'bank_trade': 5,
                'trade': 6, 'end_turn': 7
            }
            action_idx = action_map.get(action, 7)
            
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='action',
                action_idx=action_idx,
                possible_actions=possible_actions
            )
        
        return action
    
    def choose_settlement_location(self, game, valid_locations):
        location = super().choose_settlement_location(game, valid_locations)
        
        if location is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='settlement',
                action_idx=location,
                valid_options=valid_locations
            )
        
        return location
    
    def choose_city_location(self, game, valid_locations):
        location = super().choose_city_location(game, valid_locations)
        
        if location is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='city',
                action_idx=location,
                valid_options=valid_locations
            )
        
        return location
    
    def choose_road_location(self, game, valid_locations):
        location = super().choose_road_location(game, valid_locations)
        
        if location is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='road',
                action_idx=location,
                valid_options=valid_locations
            )
        
        return location
    
    def choose_dev_card_to_play(self, game, playable_cards):
        card = super().choose_dev_card_to_play(game, playable_cards)
        
        if card is not None:
            card_map = {'knight': 0, 'monopoly': 1, 'year_of_plenty': 2, 'road_building': 3, 'victory_point': 4}
            action_idx = card_map.get(card, 0)
            
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='dev_card',
                action_idx=action_idx,
                valid_options=playable_cards
            )
        
        return card
    
    def choose_bank_trade(self, game, possible_trades):
        trade = super().choose_bank_trade(game, possible_trades)
        
        if trade is not None:
            resource_map = {'wheat': 0, 'wood': 1, 'brick': 2, 'sheep': 3, 'ore': 4}
            
            give_idx = resource_map.get(trade.get('give', 'wheat'), 0)
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='exchange_give',
                action_idx=give_idx,
                valid_options=None
            )
            
            receive_idx = resource_map.get(trade.get('receive', 'wheat'), 0)
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='exchange_receive',
                action_idx=receive_idx,
                valid_options=None
            )
        
        return trade
    
    def choose_robber_placement(self, game, valid_hexes):
        hex_id = super().choose_robber_placement(game, valid_hexes)
        
        if hex_id is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='tile',
                action_idx=hex_id,
                valid_options=valid_hexes
            )
        
        return hex_id
    
    def choose_player_to_steal_from(self, game, valid_players):
        player_id = super().choose_player_to_steal_from(game, valid_players)
        
        if player_id is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='steal_player',
                action_idx=player_id,
                valid_options=valid_players
            )
        
        return player_id
    
    def choose_monopoly_resource(self, game):
        resource = super().choose_monopoly_resource(game)
        
        if resource is not None:
            resource_map = {'wheat': 0, 'wood': 1, 'brick': 2, 'sheep': 3, 'ore': 4}
            action_idx = resource_map.get(resource, 0)
            
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='resource',
                action_idx=action_idx,
                valid_options=None
            )
        
        return resource
    
    def choose_year_of_plenty_resources(self, game):
        resources = super().choose_year_of_plenty_resources(game)
        
        if resources and len(resources) >= 2:
            resource_map = {'wheat': 0, 'wood': 1, 'brick': 2, 'sheep': 3, 'ore': 4}
            
            # Log first resource
            action_idx1 = resource_map.get(resources[0], 0)
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='resource',
                action_idx=action_idx1,
                valid_options=None
            )
            
            # Log second resource
            action_idx2 = resource_map.get(resources[1], 0)
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='resource2',
                action_idx=action_idx2,
                valid_options=None
            )
        
        return resources
    
    def choose_initial_settlement(self, game, valid_locations, round_num):
        location = super().choose_initial_settlement(game, valid_locations, round_num)
        
        if location is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='settlement',
                action_idx=location,
                valid_options=valid_locations
            )
        
        return location
    
    def choose_initial_road(self, game, valid_edges, settlement_vertex, round_num):
        edge = super().choose_initial_road(game, valid_edges, settlement_vertex, round_num)
        
        if edge is not None:
            self.collector.record_decision(
                self.episode_data, game, self.id,
                head_type='road',
                action_idx=edge,
                valid_options=valid_edges
            )
        
        return edge
    
class RewardCalculator:
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.resource_types = ['ore', 'wheat', 'brick', 'wood', 'sheep']
    
    def extract_metrics(self, game, player_id):
        """Extract all relevant metrics from game state"""
        metrics = {}
        
        # vps
        metrics['vp'] = game.get_player_score(player_id)
        
        # buildings
        settlements, cities = game.get_buildings(player_id)
        metrics['settlements'] = len(settlements)
        metrics['cities'] = len(cities)
        metrics['roads'] = len([e for e in game.edges.values() if e.owner == player_id])
        
        # achievements
        metrics['longest_road'] = game.longest_road_info[0] == player_id
        metrics['largest_army'] = game.largest_army_info[0] == player_id
        
        # resources
        inventory = game.get_player_inventory(player_id)
        metrics['resources'] = sum(inventory.values())
        metrics['unique_resources'] = sum(1 for v in inventory.values() if v > 0)
        metrics['resource_counts'] = inventory.copy()
        
        # dev cards
        dev_cards = game.get_player_dev_cards(player_id)
        metrics['dev_cards'] = sum(dev_cards.values())
        metrics['dev_card_counts'] = dev_cards.copy()
        metrics['knights_played'] = game.knights_played[player_id]
        
        # point lead
        all_scores = [game.get_player_score(pid) for pid in range(game.config.num_players)]
        sorted_scores = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)
        metrics['rank'] = next(i for i, (pid, _) in enumerate(sorted_scores) if pid == player_id) + 1
        metrics['vp_lead'] = metrics['vp'] - max(s for i, s in enumerate(all_scores) if i != player_id)
        
        # ports
        metrics['ports_accessible'] = self._count_accessible_ports(game, player_id)
        
        # positioning
        metrics['high_value_settlements'] = self._count_high_value_settlements(game, settlements)
        metrics['diverse_settlements'] = self._count_diverse_settlements(game, settlements)
        
        return metrics
    
    def _count_accessible_ports(self, game, player_id):
        """Count unique ports player has access to"""
        accessible_ports = set()
        for edge_id, port_info in game.ports.items():
            edge = game.edges[edge_id]
            v1 = game.vertices[edge.vertex1]
            v2 = game.vertices[edge.vertex2]
            if v1.owner == player_id or v2.owner == player_id:
                accessible_ports.add(port_info['type'])
        return len(accessible_ports)
    
    def _count_high_value_settlements(self, game, settlements):
        """Count settlements on high-probability hexes (6 or 8)"""
        count = 0
        for settlement in settlements:
            for hex_id in settlement.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.roll_number in [6, 8]:
                    count += 1
                    break
        return count
    
    def _count_diverse_settlements(self, game, settlements):
        """Count settlements touching 3+ different resource types"""
        count = 0
        for settlement in settlements:
            resources = set()
            for hex_id in settlement.adjacent_hexes:
                hex_obj = game.hexes[hex_id]
                if hex_obj.resource != 'desert':
                    resources.add(hex_obj.resource)
            if len(resources) >= 3:
                count += 1
        return count
    
    def compute_turn_reward(self, metrics_before, metrics_after, game, player_id, 
                       action_history=None, turn_number=1):
        """
        Compute reward for a turn
        
        Args:
            metrics_before: Metrics before turn
            metrics_after: Metrics after turn
            game: Game instance
            player_id: Player ID
            action_history: List of actions taken this turn
            turn_number: Current turn number for curriculum scaling
        """
        reward = 0.0
        reward_breakdown = {}

        # vp
        vp_change = metrics_after['vp'] - metrics_before['vp']
        if vp_change > 0:
            vp_reward = self.config.vp_gain * vp_change
            reward += vp_reward
            reward_breakdown['vp_gain'] = vp_reward
        elif vp_change < 0:
            vp_penalty = self.config.vp_loss * abs(vp_change)
            reward += vp_penalty
            reward_breakdown['vp_loss'] = vp_penalty

        # buildings
        settlement_gain = metrics_after['settlements'] - metrics_before['settlements']
        if settlement_gain > 0:
            building_reward = self.config.settlement_built * settlement_gain
            reward += building_reward
            reward_breakdown['settlements'] = building_reward
        
        city_gain = metrics_after['cities'] - metrics_before['cities']
        if city_gain > 0:
            city_reward = self.config.city_built * city_gain
            reward += city_reward
            reward_breakdown['cities'] = city_reward
        
        road_gain = metrics_after['roads'] - metrics_before['roads']
        if road_gain > 0:
            road_reward = self.config.road_built * road_gain
            reward += road_reward
            reward_breakdown['roads'] = road_reward

        # longest road
        if metrics_after['longest_road'] and not metrics_before['longest_road']:
            reward += self.config.longest_road_gained
            reward_breakdown['longest_road_gained'] = self.config.longest_road_gained
        elif not metrics_after['longest_road'] and metrics_before['longest_road']:
            reward += self.config.longest_road_lost
            reward_breakdown['longest_road_lost'] = self.config.longest_road_lost
        
        # largest army
        if metrics_after['largest_army'] and not metrics_before['largest_army']:
            reward += self.config.largest_army_gained
            reward_breakdown['largest_army_gained'] = self.config.largest_army_gained
        elif not metrics_after['largest_army'] and metrics_before['largest_army']:
            reward += self.config.largest_army_lost
            reward_breakdown['largest_army_lost'] = self.config.largest_army_lost
        
        # dev cards
        dev_gain = metrics_after['dev_cards'] - metrics_before['dev_cards']
        if dev_gain > 0:
            dev_reward = self.config.dev_card_bought * dev_gain
            reward += dev_reward
            reward_breakdown['dev_cards'] = dev_reward

        knight_gain = metrics_after['knights_played'] - metrics_before['knights_played']
        if knight_gain > 0:
            knight_reward = self.config.knight_played * knight_gain
            reward += knight_reward
            reward_breakdown['knights'] = knight_reward

        vp_card_gain = metrics_after.get('vp_cards_revealed', 0) - metrics_before.get('vp_cards_revealed', 0)
        if vp_card_gain > 0:
            vp_card_reward = self.config.vp_card_revealed * vp_card_gain
            reward += vp_card_reward
            reward_breakdown['vp_cards'] = vp_card_reward
        
        # managing resources
        resource_change = metrics_after['resources'] - metrics_before['resources']
        if resource_change > 0:
            res_reward = self.config.resource_gained * resource_change
            reward += res_reward
            reward_breakdown['resources_gained'] = res_reward
        elif resource_change < 0:
            res_penalty = self.config.resource_lost * abs(resource_change)
            reward += res_penalty
            reward_breakdown['resources_spent'] = res_penalty

        if metrics_after['unique_resources'] == 5 and metrics_before['unique_resources'] < 5:
            reward += self.config.resource_diversity_bonus
            reward_breakdown['diversity'] = self.config.resource_diversity_bonus
        
        # hoarding penalty
        if metrics_after['resources'] > 7:
            excess_resources = min(metrics_after['resources'] - 7, 10)  # ⬅️ CAP at 10
            hoarding_penalty = self.config.resource_hoarding_penalty * excess_resources
            reward += hoarding_penalty
            reward_breakdown['hoarding'] = hoarding_penalty

        high_val_gain = metrics_after.get('high_value_settlements', 0) - metrics_before.get('high_value_settlements', 0)
        if high_val_gain > 0:
            pos_reward = self.config.high_value_settlement * high_val_gain
            reward += pos_reward
            reward_breakdown['high_value_pos'] = pos_reward
        
        diverse_gain = metrics_after.get('diverse_settlements', 0) - metrics_before.get('diverse_settlements', 0)
        if diverse_gain > 0:
            div_reward = self.config.diverse_settlement * diverse_gain
            reward += div_reward
            reward_breakdown['diverse_pos'] = div_reward
        
        port_gain = metrics_after.get('ports_accessible', 0) - metrics_before.get('ports_accessible', 0)
        if port_gain > 0:
            port_reward = self.config.port_access_gained * port_gain
            reward += port_reward
            reward_breakdown['port_access'] = port_reward
        
        # track robber blocking and stealing if available in metrics
        if metrics_after.get('robber_blocks', 0) > metrics_before.get('robber_blocks', 0):
            block_reward = self.config.robber_block_opponent
            reward += block_reward
            reward_breakdown['robber_block'] = block_reward
        
        if metrics_after.get('robber_steals', 0) > metrics_before.get('robber_steals', 0):
            steal_reward = self.config.robber_steal_success
            reward += steal_reward
            reward_breakdown['robber_steal'] = steal_reward

        # vp lead and deficit
        if metrics_after['vp_lead'] > 0:
            lead_reward = self.config.vp_lead_bonus * metrics_after['vp_lead']
            reward += lead_reward
            reward_breakdown['vp_lead'] = lead_reward
        elif metrics_after['vp_lead'] < 0:
            deficit_penalty = self.config.vp_behind_penalty * abs(metrics_after['vp_lead'])
            reward += deficit_penalty
            reward_breakdown['vp_deficit'] = deficit_penalty
        if action_history:
            invalid_count = 0
            trade_reward_total = 0
            
            for action_type in action_history:
                if action_type == 'trade_success':
                    trade_reward_total += self.config.efficient_trade
                elif action_type == 'port_trade':
                    trade_reward_total += self.config.port_usage
                elif action_type == 'bank_trade_4to1':
                    trade_reward_total += self.config.bank_trade_4to1
                elif action_type == 'invalid_action':
                    invalid_count += 1
            
            # cap at 5
            invalid_count = min(invalid_count, 5)
            
            if trade_reward_total != 0:
                reward += trade_reward_total
                reward_breakdown['trading'] = trade_reward_total
            
            if invalid_count > 0:
                invalid_penalty = self.config.invalid_action_penalty * invalid_count
                reward += invalid_penalty
                reward_breakdown['invalid_actions'] = invalid_penalty

        # Step penalty
        if self.config.step_penalty != 0:
            step_count = len(action_history) if action_history else 1
            step_penalty_total = self.config.step_penalty * step_count
            reward += step_penalty_total
            reward_breakdown['step_penalty'] = step_penalty_total
        
        if self.config.turn_penalty != 0:
            reward += self.config.turn_penalty
            reward_breakdown['turn_penalty'] = self.config.turn_penalty
        
        # clipping
        if self.config.normalize_rewards and self.config.reward_clip_min is not None:
            original_reward = reward
            reward = np.clip(reward, self.config.reward_clip_min, self.config.reward_clip_max)
            if reward != original_reward:
                reward_breakdown['clipped'] = f"{original_reward:.2f} → {reward:.2f}"
        
        """if abs(reward) > 100: # arbitrary check large reward
            print(f"⚠️ Large turn reward: {reward:.2f}")
            print(f"Breakdown: {reward_breakdown}")"""
        
        return reward, reward_breakdown
      
    def compute_terminal_reward(self, game, player_id):
        """Compute final reward at game end"""
        all_scores = [(pid, game.get_player_score(pid)) for pid in range(game.config.num_players)]
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        
        player_rank = next(i for i, (pid, _) in enumerate(sorted_scores) if pid == player_id) + 1
        
        if player_rank == 1:
            return self.config.win_reward
        elif player_rank == 2:
            return self.config.second_place_reward
        else:
            return self.config.loss_reward
        
def plot_training_diagnostics(network, collector, config, agent=None, game=None):
    fig = plt.figure(figsize=(20, 12))
    
    # settlement distribution
    ax1 = plt.subplot(3, 4, 1)
    settlement_choices = []
    for episode in collector.demonstrations:
        for state, action, head_type in zip(episode['states'], episode['actions'], episode['head_types']):
            if head_type == 'settlement':
                settlement_choices.append(action)
    
    if settlement_choices:
        counter = Counter(settlement_choices)
        top_20 = counter.most_common(20)
        locs, counts = zip(*top_20) if top_20 else ([], [])
        ax1.bar(range(len(locs)), counts, color='steelblue')
        ax1.set_xlabel('Settlement Location ID')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Settlement Distribution (Unique: {len(set(settlement_choices))}/54)')
        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No settlement data', ha='center', va='center')
        ax1.set_title('Settlement Distribution')

    # vertex feature variance
    ax2 = plt.subplot(3, 4, 2)
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        vertex_feats = state_dict['vertex_features'][0]
        
        feature_vars = [vertex_feats[:, i].var().item() for i in range(16)]
        ax2.bar(range(16), feature_vars, color='coral')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Variance')
        ax2.set_title('Vertex Feature Variance')
        ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax2.set_title('Vertex Feature Variance')
    
    # logit distribution
    ax3 = plt.subplot(3, 4, 3)
    if game is not None and agent is not None:
        valid_locs = game.get_valid_settlement_locations(0, initial_placement=True)
        state_dict = agent._encode_state(game)
        mask = agent._create_mask_for_head('settlement', valid_locs, None, 
                                            state_dict['action_masks'])
        
        with torch.no_grad():
            logits, value = network(state_dict, head_type='settlement', mask=mask.unsqueeze(0))
        
        logits_np = logits[0].cpu().numpy()
        ax3.hist(logits_np, bins=30, color='teal', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Logit Value')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Logits Distribution (Std: {logits.std().item():.4f})')
        ax3.axvline(logits_np.mean(), color='red', linestyle='--', alpha=0.7)
    else:
        ax3.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax3.set_title('Logits Distribution')
    
    # weight change placeholder
    ax4 = plt.subplot(3, 4, 4)
    ax4.text(0.5, 0.5, 'Weight Change Check\n(Run after training)', 
            ha='center', va='center', fontsize=11)
    ax4.set_title('Network Weight Updates')
    ax4.axis('off')
    
    # GNN output variance
    ax5 = plt.subplot(3, 4, 5)
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        with torch.no_grad():
            vertex_out = network._process_vertices(state_dict)
        
        vertex_norms = [vertex_out[0, i].norm().item() for i in range(min(54, vertex_out.shape[1]))]
        ax5.plot(vertex_norms, marker='o', markersize=3, linewidth=1, color='purple')
        ax5.set_xlabel('Vertex Index')
        ax5.set_ylabel('Embedding Norm')
        ax5.set_title(f'GNN Output Norms (Var: {vertex_out.var().item():.6f})')
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax5.set_title('GNN Output Variance')

    # data diversity
    ax6 = plt.subplot(3, 4, 6)
    settlement_actions = []
    num_valid_list = []
    for episode in collector.demonstrations:
        for action, head_type, mask in zip(episode['actions'], episode['head_types'], 
                                           episode['masks']):
            if head_type == 'settlement':
                settlement_actions.append(action)
                num_valid_list.append(mask.sum() if hasattr(mask, 'sum') else sum(mask))
    
    if settlement_actions:
        unique_actions = len(set(settlement_actions))
        ax6.text(0.5, 0.65, f'Total: {len(settlement_actions)}', 
                ha='center', fontsize=11, transform=ax6.transAxes)
        ax6.text(0.5, 0.5, f'Unique: {unique_actions}', 
                ha='center', fontsize=12, transform=ax6.transAxes, fontweight='bold')
        ax6.text(0.5, 0.35, f'Coverage: {unique_actions/54*100:.1f}%', 
                ha='center', fontsize=11, transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'No settlement data', ha='center', va='center')
    
    ax6.set_title('Training Data Diversity')
    ax6.axis('off')

    # action type distribution
    ax7 = plt.subplot(3, 4, 7)
    action_types = []
    for episode in collector.demonstrations:
        action_types.extend(episode['head_types'])
    
    if action_types:
        type_counter = Counter(action_types)
        types, counts = zip(*sorted(type_counter.items(), key=lambda x: -x[1]))
        ax7.barh(range(len(types)), counts, color='lightcoral')
        ax7.set_yticks(range(len(types)))
        ax7.set_yticklabels(types, fontsize=8)
        ax7.set_xlabel('Count')
        ax7.set_title('Action Type Distribution')
        ax7.grid(axis='x', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No action data', ha='center', va='center')
        ax7.set_title('Action Type Distribution')

    # network output
    ax8 = plt.subplot(3, 4, 8)
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        
        health_checks = []
        with torch.no_grad():
            mask = torch.ones(1, 8, device=agent.device)
            logits, value = network(state_dict, head_type='action', mask=mask)
            
            has_nan = torch.isnan(logits).any() or torch.isnan(value).any()
            has_inf = torch.isinf(logits).any() or torch.isinf(value).any()
            
            health_checks.append(('NaN Free', not has_nan))
            health_checks.append(('Inf Free', not has_inf))
            health_checks.append(('Logits OK', logits.min() > -100 and logits.max() < 100))
            health_checks.append(('Value OK', value.abs() < 1000))
        
        y_pos = 0.75
        for check_name, passed in health_checks:
            symbol = '[OK]' if passed else '[FAIL]'
            color = 'green' if passed else 'red'
            ax8.text(0.1, y_pos, f'{symbol} {check_name}', 
                    transform=ax8.transAxes, fontsize=10, color=color, fontweight='bold')
            y_pos -= 0.18
        
        ax8.set_title('Network Output Health')
        ax8.axis('off')
    else:
        ax8.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax8.set_title('Network Output Health')
        ax8.axis('off')

    # vertex feature stats
    ax9 = plt.subplot(3, 4, 9)
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        vertex_feats = state_dict['vertex_features'][0]
        
        stats_text = f"Min:   {vertex_feats.min():.4f}\n"
        stats_text += f"Max:   {vertex_feats.max():.4f}\n"
        stats_text += f"Mean:  {vertex_feats.mean():.4f}\n"
        stats_text += f"Std:   {vertex_feats.std():.4f}\n"
        stats_text += f"Shape: {tuple(vertex_feats.shape)}"
        
        ax9.text(0.1, 0.7, stats_text, transform=ax9.transAxes, 
                fontsize=9, family='monospace', verticalalignment='top')
        ax9.set_title('Vertex Feature Stats')
        ax9.axis('off')
    else:
        ax9.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax9.set_title('Vertex Feature Stats')
        ax9.axis('off')

    # state encoding shapes
    ax10 = plt.subplot(3, 4, 10)
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        
        shapes_text = "State Shapes:\n\n"
        for key in ['hex_features', 'vertex_features', 'edge_features', 
                    'player_vector', 'global_context']:
            if key in state_dict:
                shape = state_dict[key].shape
                shapes_text += f"{key}:\n  {tuple(shape)}\n"
        
        ax10.text(0.05, 0.95, shapes_text, transform=ax10.transAxes, 
                 fontsize=8, family='monospace', verticalalignment='top')
        ax10.set_title('State Encoding Shapes')
        ax10.axis('off')
    else:
        ax10.text(0.5, 0.5, 'Need game & agent', ha='center', va='center')
        ax10.set_title('State Encoding Shapes')
        ax10.axis('off')

    # masking
    ax11 = plt.subplot(3, 4, 11)
    if settlement_actions and num_valid_list:
        ax11.hist(num_valid_list, bins=20, color='olive', alpha=0.7, edgecolor='black')
        ax11.set_xlabel('Valid Options')
        ax11.set_ylabel('Frequency')
        ax11.set_title(f'Valid Options per Decision (Mean: {np.mean(num_valid_list):.1f})')
        ax11.grid(axis='y', alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'No mask data', ha='center', va='center')
        ax11.set_title('Mask Coverage')
        
    # Summary
    ax12 = plt.subplot(3, 4, 12)
    
    summary_text = "Summary\n" + "="*30 + "\n\n"
    
    issues = []
    
    if settlement_choices and len(set(settlement_choices)) < 10:
        issues.append("Low settlement diversity")
    
    if game is not None and agent is not None:
        state_dict = agent._encode_state(game)
        vertex_feats = state_dict['vertex_features'][0]
        
        if vertex_feats.std(dim=0).max() < 0.01:
            issues.append("Identical vertex features")
        
        valid_locs = game.get_valid_settlement_locations(0, initial_placement=True)
        mask = agent._create_mask_for_head('settlement', valid_locs, None, 
                                            state_dict['action_masks'])
        with torch.no_grad():
            logits, _ = network(state_dict, head_type='settlement', mask=mask.unsqueeze(0))
        
        if logits.std().item() < 0.01:
            issues.append("No logit variation")
    
    if len(settlement_actions) > 0 and len(set(settlement_actions)) < 10:
        issues.append("Low training diversity")
    
    if issues:
        summary_text += "issues:\n"
        for issue in issues:
            summary_text += f"  - {issue}\n"
    else:
        summary_text += "[OK] All checks passed\n"
    
    summary_text += f"\n{'='*30}\n"
    summary_text += f"Demonstrations: {len(collector.demonstrations)}\n"
    summary_text += f"Total decisions: {sum(len(ep['actions']) for ep in collector.demonstrations)}"
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
             fontsize=9, verticalalignment='top', family='monospace')
    ax12.set_title('Status Summary')
    ax12.axis('off')
    
    plt.tight_layout()
    return fig

class ImitationLearningTrainer:
    """Train RL agent on GreedyAgent demonstrations"""
    def __init__(self, network, lr=3e-4):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using metal performance shaders")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        self.network.to(self.device)
        
        if self.device.type == 'mps':
            torch.backends.mps.enable_jit = False
    
    def train(self, demonstrations, num_epochs=50, batch_size=32, 
              balance_heads=True, val_split=0.15):
        self.network.train()
        for param in self.network.parameters():
            param.requires_grad = True
        
        print(f"\nTraining on {len(demonstrations)} games")
        print(f"Parameters: {sum(p.numel() for p in self.network.parameters() if p.requires_grad):,}")
        
        # Train/val split
        if val_split > 0:
            num_val = max(1, int(len(demonstrations) * val_split))
            indices = np.random.permutation(len(demonstrations))
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]
            
            train_demos = [demonstrations[i] for i in train_indices]
            val_demos = [demonstrations[i] for i in val_indices]
            
            print(f"  Training: {len(train_demos)} games")
            print(f"  Validation: {len(val_demos)} games ({val_split:.0%})")
        else:
            train_demos = demonstrations
            val_demos = None
            print(f"  No validation split")
        
        train_data_by_head = self._organize_data_by_head(train_demos)
        
        if val_demos is not None:
            val_data_by_head = self._organize_data_by_head(val_demos)
        else:
            val_data_by_head = None
        
        # dataset stats
        print("\nTraining Dataset:")
        head_counts = {h: len(data['states']) for h, data in train_data_by_head.items()}
        total_samples = sum(head_counts.values())

        for head_type, count in head_counts.items():
            pct = (count / total_samples) * 100 if total_samples > 0 else 0.0
            print(f"  {head_type:20s}: {count:6d} ({pct:5.2f}%)")
        print(f"  {'Total':20s}: {total_samples:6d}")
        
        if val_data_by_head is not None:
            print("\nValidation Dataset:")
            val_head_counts = {h: len(data['states']) for h, data in val_data_by_head.items()}
            val_total = sum(val_head_counts.values())
            for head_type, count in val_head_counts.items():
                pct = (count / val_total) * 100 if val_total > 0 else 0.0
                print(f"  {head_type:20s}: {count:6d} ({pct:5.2f}%)")
            print(f"  {'Total':20s}: {val_total:6d}")
        
        if balance_heads:
            total = sum(head_counts.values())
            proportions = {h: c / total for h, c in head_counts.items()}
            raw_w = {h: 1.0 / np.sqrt(p) for h, p in proportions.items()}
            mean_w = np.mean(list(raw_w.values()))
            head_weights = {h: w / mean_w for h, w in raw_w.items()}
            
            print("\nHead Balancing:")
            for head_type, weight in head_weights.items():
                print(f"  {head_type:20s}: {weight:.2f}x")
        else:
            head_weights = {h: 1.0 for h in head_counts}
            
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'train_loss_by_head': defaultdict(list),
            'train_accuracy_by_head': defaultdict(list),
            'val_loss_by_head': defaultdict(list),
            'val_accuracy_by_head': defaultdict(list),
        }

        best_val_acc = 0.0
        best_epoch = 0

        print("\nStarting training\n")
        for epoch in tqdm(range(num_epochs), desc="Training"):
            
            train_loss, train_acc = self._train_epoch(
                train_data_by_head, head_weights, batch_size, metrics
            )
            
            metrics['train_loss'].append(train_loss)
            metrics['train_accuracy'].append(train_acc)
            
            if val_data_by_head is not None:
                val_loss, val_acc, val_acc_by_head, val_loss_by_head = self._validate(
                    val_data_by_head, batch_size
                )
                
                metrics['val_loss'].append(val_loss)
                metrics['val_accuracy'].append(val_acc)
                
                for head_type, acc in val_acc_by_head.items():
                    metrics['val_accuracy_by_head'][head_type].append(acc)
                for head_type, loss in val_loss_by_head.items():
                    metrics['val_loss_by_head'][head_type].append(loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    improvement = "[NEW BEST]"
                else:
                    improvement = ""
            
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"{'='*70}")
                
                if val_data_by_head is not None:
                    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}  |  "
                        f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2%} {improvement}")
                else:
                    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
                
                print(f"\nPer-Head:")
                if val_data_by_head is not None:
                    print(f"{'Head':<20} {'Tr.Loss':>10} {'Tr.Acc':>10} {'Val.Loss':>10} {'Val.Acc':>10}")
                    print(f"{'-'*62}")
                else:
                    print(f"{'Head':<20} {'Train Loss':>12} {'Train Acc':>12}")
                    print(f"{'-'*46}")
                
                for head in sorted(train_data_by_head.keys()):
                    if head in metrics['train_loss_by_head'] and metrics['train_loss_by_head'][head]:
                        train_head_loss = np.mean(metrics['train_loss_by_head'][head][-10:])
                        train_loss_str = f"{train_head_loss:10.4f}"
                    else:
                        train_loss_str = f"{'--':>10}"
                    
                    if head in metrics['train_accuracy_by_head'] and metrics['train_accuracy_by_head'][head]:
                        train_head_acc = metrics['train_accuracy_by_head'][head][-1]
                        train_acc_str = f"{train_head_acc:9.1%}"
                    else:
                        train_acc_str = f"{'--':>10}"
                    
                    if val_data_by_head is not None:
                        if head in val_loss_by_head:
                            val_loss_str = f"{val_loss_by_head[head]:10.4f}"
                        else:
                            val_loss_str = f"{'--':>10}"
                        
                        if head in val_acc_by_head:
                            val_acc_str = f"{val_acc_by_head[head]:9.1%}"
                        else:
                            val_acc_str = f"{'--':>10}"
                        
                        print(f"{head:<20} {train_loss_str} {train_acc_str} {val_loss_str} {val_acc_str}")
                    else:
                        print(f"{head:<20} {train_loss_str} {train_acc_str}")
        
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"{'='*60}")
        print(f"  Final train loss: {metrics['train_loss'][-1]:.4f}")
        print(f"  Final train accuracy: {metrics['train_accuracy'][-1]:.2%}")
        
        if val_data_by_head is not None:
            print(f"  Final val loss: {metrics['val_loss'][-1]:.4f}")
            print(f"  Final val accuracy: {metrics['val_accuracy'][-1]:.2%}")
            print(f"  Best val accuracy: {best_val_acc:.2%} (epoch {best_epoch})")
            
            if metrics['val_accuracy'][-1] < best_val_acc - 0.03:
                print(f"  Warning: possible overfitting")
        
        return metrics
    
    def _organize_data_by_head(self, demonstrations):
        """Organize demonstration data by action head type"""
        data_by_head = defaultdict(lambda: {'states': [], 'actions': [], 'masks': []})
        
        print("Transferring data to device...")
        for episode in tqdm(demonstrations, desc="  Processing"):
            for state, action, head_type, mask in zip(
                episode['states'], episode['actions'],
                episode['head_types'], episode['masks']
            ):
                state_tensors = {}
                for field in ['hex_features', 'vertex_features', 'edge_features',
                             'player_vector', 'global_context']:
                    if isinstance(state[field], np.ndarray):
                        state_tensors[field] = torch.from_numpy(state[field]).float().to(self.device)
                    else:
                        state_tensors[field] = state[field].float().to(self.device)
                
                data_by_head[head_type]['states'].append(state_tensors)
                data_by_head[head_type]['actions'].append(action)
                data_by_head[head_type]['masks'].append(mask)
        
        return data_by_head
    
    def _train_epoch(self, data_by_head, head_weights, batch_size, metrics):
        self.network.train()
        
        epoch_losses = []
        epoch_accuracies = []
        head_accuracies = defaultdict(list)
        
        for head_type in data_by_head.keys():
            data = data_by_head[head_type]
            if not data['states']:
                continue
            
            num_samples = len(data['states'])
            weight = head_weights[head_type]
            effective_samples = int(num_samples * weight)
            
            if weight > 1.0:
                indices = torch.randint(0, num_samples, (effective_samples,))
            else:
                indices = torch.randperm(num_samples)[:effective_samples]
            
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_idx = indices[start:end]
                
                batch_states = [data['states'][i] for i in batch_idx]
                batched_state = self._batch_state_dicts(batch_states)
                
                batch_actions = torch.tensor(
                    [data['actions'][i] for i in batch_idx],
                    device=self.device, dtype=torch.long
                )
                
                batch_masks_list = [data['masks'][i] for i in batch_idx]
                
                if any(m is None for m in batch_masks_list):
                    batch_masks = None
                else:
                    batch_masks = torch.tensor(
                        np.array(batch_masks_list),
                        device=self.device, dtype=torch.float32
                    )
                
                logits, _ = self.network(batched_state, head_type=head_type, mask=None)
                loss = F.cross_entropy(logits, batch_actions)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nInvalid loss for {head_type}, skipping batch")
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                metrics['train_loss_by_head'][head_type].append(loss.item())
                
                with torch.no_grad():
                    if batch_masks is not None:
                        masked_logits = logits.clone()
                        masked_logits[batch_masks == 0] = -1e9
                        preds = masked_logits.argmax(dim=-1)
                    else:
                        preds = logits.argmax(dim=-1)
                    
                    accuracy = (preds == batch_actions).float().mean().item()
                    epoch_accuracies.append(accuracy)
                    head_accuracies[head_type].append(accuracy)
        
        for head_type, accs in head_accuracies.items():
            metrics['train_accuracy_by_head'][head_type].append(np.mean(accs))
        
        return np.mean(epoch_losses) if epoch_losses else 0.0, \
               np.mean(epoch_accuracies) if epoch_accuracies else 0.0
    
    def _validate(self, val_data_by_head, batch_size):
        self.network.eval()
        
        val_losses = []
        val_accuracies = []
        val_acc_by_head = {}
        val_loss_by_head = {}
        
        with torch.no_grad():
            for head_type, data in val_data_by_head.items():
                if not data['states']:
                    continue
                
                head_losses = []
                head_accuracies = []
                
                num_samples = len(data['states'])
                
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    
                    batch_states = data['states'][start:end]
                    batched_state = self._batch_state_dicts(batch_states)
                    
                    batch_actions = torch.tensor(
                        data['actions'][start:end],
                        device=self.device, dtype=torch.long
                    )
                    
                    batch_masks_list = data['masks'][start:end]
                    if any(m is None for m in batch_masks_list):
                        batch_masks = None
                    else:
                        batch_masks = torch.tensor(
                            np.array(batch_masks_list),
                            device=self.device, dtype=torch.float32
                        )
                    
                    logits, _ = self.network(batched_state, head_type=head_type, mask=None)
                    
                    loss = F.cross_entropy(logits, batch_actions)
                    head_losses.append(loss.item())
                    val_losses.append(loss.item())
                    
                    if batch_masks is not None:
                        masked_logits = logits.clone()
                        masked_logits[batch_masks == 0] = -1e9
                        preds = masked_logits.argmax(dim=-1)
                    else:
                        preds = logits.argmax(dim=-1)
                    
                    accuracy = (preds == batch_actions).float().mean().item()
                    head_accuracies.append(accuracy)
                    val_accuracies.append(accuracy)
                
                if head_accuracies:
                    val_acc_by_head[head_type] = np.mean(head_accuracies)
                if head_losses:
                    val_loss_by_head[head_type] = np.mean(head_losses)
        
        return np.mean(val_losses) if val_losses else 0.0, \
               np.mean(val_accuracies) if val_accuracies else 0.0, \
               val_acc_by_head, val_loss_by_head
    
    def _batch_state_dicts(self, state_dict_list):
        """Batch state dicts into one dict w/ batched tensors"""
        if not state_dict_list:
            return None
        
        batched = {}
        tensor_fields = ['hex_features', 'vertex_features', 'edge_features',
                        'player_vector', 'global_context']
        
        for field in tensor_fields:
            values = [sd[field] for sd in state_dict_list]
            
            tensors = []
            for val in values:
                if isinstance(val, np.ndarray):
                    t = torch.from_numpy(val).float().to(self.device)
                else:
                    t = val.to(self.device).float()
                
                if field in ['hex_features', 'vertex_features', 'edge_features']:
                    if t.dim() == 2:
                        pass
                    elif t.dim() == 3:
                        t = t.squeeze(0)
                    else:
                        raise ValueError(f"{field} unexpected shape: {t.shape}")
                    t = t.unsqueeze(0)
                
                elif field in ['player_vector', 'global_context']:
                    if t.dim() == 1:
                        pass
                    elif t.dim() == 2:
                        t = t.squeeze(0)
                    else:
                        raise ValueError(f"{field} unexpected shape: {t.shape}")
                    t = t.unsqueeze(0)
                
                tensors.append(t)
            
            batched[field] = torch.cat(tensors, dim=0)
        
        batched['action_masks'] = [sd.get('action_masks') for sd in state_dict_list]
        
        if 'graph_edges' in state_dict_list[0]:
            graph_edges = state_dict_list[0]['graph_edges']
            if graph_edges is not None:
                if isinstance(graph_edges, np.ndarray):
                    batched['graph_edges'] = torch.from_numpy(graph_edges).long().to(self.device)
                else:
                    batched['graph_edges'] = graph_edges.to(self.device)
        
        if 'edge_types' in state_dict_list[0]:
            edge_types = state_dict_list[0]['edge_types']
            if edge_types is not None:
                if isinstance(edge_types, np.ndarray):
                    batched['edge_types'] = torch.from_numpy(edge_types).long().to(self.device)
                else:
                    batched['edge_types'] = edge_types.to(self.device)
        
        return batched
    
def evaluate_agent(network, opponent_class, config, num_games=20):
    agent1 = CatanRLAgent(
        id=0, 
        color='red', 
        config=config,
        network=network,
        temperature=0.1
    )
    
    if opponent_class == CatanRLAgent:
        # self-play
        agent2 = CatanRLAgent(
            id=1,
            color='blue',
            config=config,
            network=network,
            temperature=0.1
        )
        icon = "🤖"
    else:
        agent2 = opponent_class(1, 'blue')
        icon = "🎲" if opponent_class == RandomAgent else "🧠"
    
    print(f"\n{icon} vs {opponent_class.__name__}:")
    
    players = [agent1, agent2]
    config.num_players = len(players)
    
    wins = {0: 0, 1: 0}
    vps = {0: [], 1: []}
    times = []
    
    for _ in tqdm(range(num_games), desc="  Playing"):
        game = CatanGame(config=config, seed=random.randint(0, 10000), players=players)
        t1 = time.time()
        winner = game.play_game()
        t2 = time.time()
        
        times.append(t2 - t1)
        wins[winner] += 1
        
        for i in range(2):
            vps[i].append(game.get_player_score(i))
    
    rl_win_rate = wins[0] / num_games * 100
    print(f"  Win rate: {rl_win_rate:.1f}% ({wins[0]}/{num_games})")
    print(f"  Avg VPs: RL={np.mean(vps[0]):.1f}, {opponent_class.__name__}={np.mean(vps[1]):.1f}")
    print(f"  Avg time: {np.mean(times):.2f}s")
    
    return wins[0] / num_games

def plot_imitation_training_metrics(
    metrics: Dict,
    model_config: Dict,
    training_config: Dict,
    total_params: int,
    num_demonstrations: int,
    teacher_agent_type: str = "GreedyAgent",
    save_path: Optional[str] = None,
    show: bool = True
):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    train_color = '#2E86AB'
    val_color = '#A23B72'

    fig.suptitle(
        f'Catan Imitation Learning Training\n'
        f'GNN-Transformer Network ({total_params:,} parameters) | '
        f'Teacher: {teacher_agent_type} | '
        f'{num_demonstrations:,} demonstrations',
        fontsize=16, fontweight='bold', y=0.985
    )

    ax_loss = fig.add_subplot(gs[0:2, 0])
    epochs = np.arange(1, len(metrics['train_loss']) + 1)

    ax_loss.plot(epochs, metrics['train_loss'],
                 label='Training', linewidth=2.5, color=train_color, alpha=0.8)
    ax_loss.fill_between(epochs, metrics['train_loss'], alpha=0.2, color=train_color)

    if metrics.get('val_loss') and len(metrics['val_loss']) > 0:
        ax_loss.plot(epochs, metrics['val_loss'],
                     label='Validation', linewidth=2.5, color=val_color,
                     alpha=0.8, marker='o', markersize=4,
                     markevery=max(1, len(epochs)//20))
        ax_loss.fill_between(epochs, metrics['val_loss'], alpha=0.2, color=val_color)

    ax_loss.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax_loss.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax_loss.set_title('Training Progress: Loss', fontsize=15, fontweight='bold', pad=15)
    ax_loss.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    ax_loss.set_xlim(1, len(epochs))
    ax_loss.tick_params(labelsize=11)

    if metrics.get('val_loss') and len(metrics['val_loss']) > 0:
        best_val_idx = np.argmin(metrics['val_loss'])
        best_val_loss = metrics['val_loss'][best_val_idx]
        ax_loss.axhline(y=best_val_loss, color=val_color, linestyle=':', alpha=0.5, linewidth=1.5)
        ax_loss.text(len(epochs) * 0.98, best_val_loss,
                     f'Best: {best_val_loss:.4f}',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=val_color, alpha=0.2))

    ax_acc = fig.add_subplot(gs[0:2, 1])
    ax_acc.plot(epochs, np.array(metrics['train_accuracy']) * 100,
                label='Training', linewidth=2.5, color=train_color, alpha=0.8)
    ax_acc.fill_between(epochs, np.array(metrics['train_accuracy']) * 100,
                        alpha=0.2, color=train_color)

    if metrics.get('val_accuracy') and len(metrics['val_accuracy']) > 0:
        ax_acc.plot(epochs, np.array(metrics['val_accuracy']) * 100,
                    label='Validation', linewidth=2.5, color=val_color,
                    alpha=0.8, marker='o', markersize=4,
                    markevery=max(1, len(epochs)//20))
        ax_acc.fill_between(epochs, np.array(metrics['val_accuracy']) * 100,
                            alpha=0.2, color=val_color)

    ax_acc.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax_acc.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax_acc.set_title('Training Progress: Accuracy', fontsize=15, fontweight='bold', pad=15)
    ax_acc.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    ax_acc.set_xlim(1, len(epochs))
    ax_acc.set_ylim(0, 100)
    ax_acc.tick_params(labelsize=11)

    if metrics.get('val_accuracy') and len(metrics['val_accuracy']) > 0:
        best_val_idx = np.argmax(metrics['val_accuracy'])
        best_val_acc = metrics['val_accuracy'][best_val_idx] * 100
        ax_acc.axhline(y=best_val_acc, color=val_color, linestyle=':', alpha=0.5, linewidth=1.5)
        ax_acc.text(len(epochs) * 0.98, best_val_acc,
                    f'Best: {best_val_acc:.2f}%',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=val_color, alpha=0.2))

    ax_lr = fig.add_subplot(gs[0, 2])
    if 'learning_rates' in metrics and metrics['learning_rates']:
        ax_lr.plot(epochs, metrics['learning_rates'],
                   linewidth=2, color='#E63946', alpha=0.8)
        ax_lr.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_lr.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax_lr.set_title('LR Schedule', fontsize=13, fontweight='bold', pad=10)
        ax_lr.grid(True, alpha=0.3, linestyle='--')
        ax_lr.set_yscale('log')
        ax_lr.tick_params(labelsize=10)
    else:
        lr = training_config.get('lr', 3e-4)
        ax_lr.axhline(y=lr, color='#E63946', linewidth=2, alpha=0.8)
        ax_lr.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_lr.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax_lr.set_title('LR Schedule (Constant)', fontsize=13, fontweight='bold', pad=10)
        ax_lr.grid(True, alpha=0.3, linestyle='--')
        ax_lr.set_yscale('log')
        ax_lr.set_xlim(1, len(epochs))
        ax_lr.tick_params(labelsize=10)
        ax_lr.text(len(epochs)/2, lr, f'LR = {lr:.2e}',
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#E63946', alpha=0.2))

    ax_grad = fig.add_subplot(gs[1, 2])
    if 'grad_norms' in metrics and metrics['grad_norms']:
        ax_grad.plot(epochs, metrics['grad_norms'],
                     linewidth=2, color='#06A77D', alpha=0.8)
        ax_grad.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax_grad.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_grad.set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
        ax_grad.set_title('Gradient Magnitudes', fontsize=13, fontweight='bold', pad=10)
        ax_grad.grid(True, alpha=0.3, linestyle='--')
        ax_grad.set_yscale('log')
        ax_grad.set_xlim(1, len(epochs))
        ax_grad.tick_params(labelsize=10)
    else:
        ax_grad.text(0.5, 0.5, 'Gradient norms\nnot tracked',
                     ha='center', va='center', fontsize=12, color='gray',
                     transform=ax_grad.transAxes)
        ax_grad.set_xticks([])
        ax_grad.set_yticks([])
        for side in ['top', 'right', 'bottom', 'left']:
            ax_grad.spines[side].set_visible(False)

    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.axis('off')

    info_text = f"""NETWORK ARCHITECTURE
    {'─' * 38}
    Hidden Dim:         {model_config.get('hidden_dim', 'N/A')}
    GNN Layers:         {model_config.get('num_gnn_layers', 'N/A')}
    Attention Heads:    {model_config.get('num_attention_heads', 'N/A')}
    Dropout:            {model_config.get('dropout', 'N/A')}
    Trunk Hidden:       {model_config.get('trunk_hidden', 'N/A')}
    Total Parameters:   {total_params:,}
    """
    ax_info.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
                 verticalalignment='center', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

    ax_train_info = fig.add_subplot(gs[2, 1])
    ax_train_info.axis('off')

    train_info_text = f"""TRAINING CONFIGURATION
{'─' * 38}
Teacher Agent:      {teacher_agent_type}
Demonstrations:     {num_demonstrations:,}
Batch Size:         {training_config.get('batch_size', 'N/A')}
Learning Rate:      {training_config.get('lr', 'N/A'):.2e}
Epochs:             {training_config.get('num_epochs', 'N/A')}
Val Split:          {training_config.get('val_split', 'N/A')*100:.0f}%
Balance Heads:      {training_config.get('balance_heads', 'N/A')}
"""
    ax_train_info.text(0.05, 0.5, train_info_text, fontsize=11, family='monospace',
                       verticalalignment='center', transform=ax_train_info.transAxes,
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.3))

    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis('off')

    final_train_loss = metrics['train_loss'][-1]
    final_train_acc = metrics['train_accuracy'][-1] * 100

    if metrics.get('val_loss') and len(metrics['val_loss']) > 0:
        final_val_loss = metrics['val_loss'][-1]
        final_val_acc = metrics['val_accuracy'][-1] * 100
        best_val_loss = min(metrics['val_loss'])
        best_val_acc = max(metrics['val_accuracy']) * 100
    else:
        final_val_loss = final_val_acc = best_val_loss = best_val_acc = None

    summary_lines = [
        "FINAL METRICS",
        "─" * 38,
        f"Train Loss:         {final_train_loss:.4f}",
        f"Train Acc:          {final_train_acc:.2f}%",
    ]

    if final_val_loss is not None:
        summary_lines.extend([
            "",
            f"Val Loss:           {final_val_loss:.4f}",
            f"Val Acc:            {final_val_acc:.2f}%",
            "",
            f"Best Val Acc:       {best_val_acc:.2f}%",
            f"@ Loss:             {best_val_loss:.4f}",
        ])

    summary_text = "\n".join(summary_lines)

    ax_summary.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=ax_summary.transAxes,
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved training visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig