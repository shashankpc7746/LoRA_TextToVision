"""
Cache Manager for Task 4 Day 2
Manages caching of backgrounds, poses, seeds, and features for reuse
"""

import os
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    timestamp: float
    hits: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CacheManager:
    """Intelligent caching system for video generation assets"""

    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_index: Dict[str, CacheEntry] = {}
        self._load_index()

        # Cache subdirectories
        self.backgrounds_dir = self.cache_dir / "backgrounds"
        self.poses_dir = self.cache_dir / "poses"
        self.seeds_dir = self.cache_dir / "seeds"
        self.features_dir = self.cache_dir / "features"

        for dir_path in [self.backgrounds_dir, self.poses_dir, self.seeds_dir, self.features_dir]:
            dir_path.mkdir(exist_ok=True)

    def _load_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.cache_index[key] = CacheEntry(**entry_data)
            except Exception as e:
                print(f"Warning: Failed to load cache index: {e}")

    def _save_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "index.json"
        try:
            data = {k: asdict(v) for k, v in self.cache_index.items()}
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache index: {e}")

    def _get_cache_key(self, content: str, prefix: str = "") -> str:
        """Generate cache key from content"""
        return f"{prefix}_{hashlib.md5(content.encode()).hexdigest()[:16]}"

    def _evict_if_needed(self):
        """Evict old entries if cache size exceeds limit"""
        total_size = sum(entry.size_bytes for entry in self.cache_index.values())

        if total_size > self.max_size_bytes:
            # Sort by hits (ascending) then by timestamp (ascending)
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: (x[1].hits, x[1].timestamp)
            )

            # Remove least used entries until under limit
            for key, entry in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Keep 80% utilization
                    break

                # Remove from disk
                cache_file = self._get_cache_path(key, entry.metadata.get('type', 'misc'))
                if cache_file.exists():
                    cache_file.unlink()

                total_size -= entry.size_bytes
                del self.cache_index[key]

            self._save_index()

    def _get_cache_path(self, key: str, cache_type: str) -> Path:
        """Get cache file path for given key and type"""
        type_dirs = {
            'background': self.backgrounds_dir,
            'pose': self.poses_dir,
            'seed': self.seeds_dir,
            'feature': self.features_dir,
            'misc': self.cache_dir
        }
        return type_dirs.get(cache_type, self.cache_dir) / f"{key}.pkl"

    def put(self, key: str, data: Any, metadata: Dict[str, Any] = None) -> str:
        """Store data in cache"""
        if metadata is None:
            metadata = {}

        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            metadata=metadata
        )

        # Estimate size
        try:
            entry.size_bytes = len(pickle.dumps(data))
        except:
            entry.size_bytes = 1024  # Default estimate

        # Store on disk
        cache_file = self._get_cache_path(key, metadata.get('type', 'misc'))
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to cache {key}: {e}")
            return key

        # Update index
        self.cache_index[key] = entry
        self._save_index()

        # Evict if needed
        self._evict_if_needed()

        return key

    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache"""
        if key not in self.cache_index:
            return None

        entry = self.cache_index[key]

        # Load from disk
        cache_file = self._get_cache_path(key, entry.metadata.get('type', 'misc'))
        if not cache_file.exists():
            # Remove from index if file missing
            del self.cache_index[key]
            self._save_index()
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Update hit count
            entry.hits += 1
            self._save_index()

            return data
        except Exception as e:
            print(f"Warning: Failed to load cached {key}: {e}")
            return None

    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache_index

    def get_or_create(self, key: str, creator_func, metadata: Dict[str, Any] = None) -> Any:
        """Get from cache or create and cache"""
        cached = self.get(key)
        if cached is not None:
            return cached

        # Create new
        data = creator_func()

        # Cache it
        self.put(key, data, metadata)

        return data

    # Specific cache methods for different asset types

    def cache_background(self, scene_type: str, style: str, background_data: Any) -> str:
        """Cache background asset"""
        key = self._get_cache_key(f"bg_{scene_type}_{style}", "background")
        metadata = {
            'type': 'background',
            'scene_type': scene_type,
            'style': style
        }
        return self.put(key, background_data, metadata)

    def get_background(self, scene_type: str, style: str) -> Optional[Any]:
        """Get cached background"""
        key = self._get_cache_key(f"bg_{scene_type}_{style}", "background")
        return self.get(key)

    def cache_pose(self, pose_name: str, pose_data: Any) -> str:
        """Cache pose asset"""
        key = self._get_cache_key(f"pose_{pose_name}", "pose")
        metadata = {
            'type': 'pose',
            'pose_name': pose_name
        }
        return self.put(key, pose_data, metadata)

    def get_pose(self, pose_name: str) -> Optional[Any]:
        """Get cached pose"""
        key = self._get_cache_key(f"pose_{pose_name}", "pose")
        return self.get(key)

    def cache_seed(self, prompt_hash: str, seed_data: Dict[str, Any]) -> str:
        """Cache seed and features for keyframes"""
        key = self._get_cache_key(f"seed_{prompt_hash}", "seed")
        metadata = {
            'type': 'seed',
            'prompt_hash': prompt_hash
        }
        return self.put(key, seed_data, metadata)

    def get_seed(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached seed"""
        key = self._get_cache_key(f"seed_{prompt_hash}", "seed")
        return self.get(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache_index)
        total_size = sum(entry.size_bytes for entry in self.cache_index.values())
        total_hits = sum(entry.hits for entry in self.cache_index.values())

        type_counts = {}
        for entry in self.cache_index.values():
            cache_type = entry.metadata.get('type', 'misc')
            type_counts[cache_type] = type_counts.get(cache_type, 0) + 1

        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'total_hits': total_hits,
            'type_breakdown': type_counts,
            'hit_rate': total_hits / max(total_entries, 1)
        }

    def clear_cache(self, cache_type: str = None):
        """Clear cache entries"""
        if cache_type:
            # Clear specific type
            keys_to_remove = [
                key for key, entry in self.cache_index.items()
                if entry.metadata.get('type') == cache_type
            ]
            for key in keys_to_remove:
                del self.cache_index[key]
        else:
            # Clear all
            self.cache_index.clear()

        self._save_index()

        # Remove files
        import shutil
        if cache_type:
            type_dirs = {
                'background': self.backgrounds_dir,
                'pose': self.poses_dir,
                'seed': self.seeds_dir,
                'feature': self.features_dir
            }
            if cache_type in type_dirs:
                shutil.rmtree(type_dirs[cache_type], ignore_errors=True)
                type_dirs[cache_type].mkdir(exist_ok=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(exist_ok=True)
            for subdir in [self.backgrounds_dir, self.poses_dir, self.seeds_dir, self.features_dir]:
                subdir.mkdir(exist_ok=True)


# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager