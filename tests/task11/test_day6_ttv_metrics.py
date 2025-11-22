"""
Test Day 6: TTV Intelligence Metrics Logging

Tests the integration of TTV Studio intelligence metrics into the existing
audit_logger system from Task 10.

Coverage:
- Story analysis metrics
- Scene graph metrics
- Narrative metrics
- Emotion metrics
- Extension metrics
- Quality metrics
- Performance metrics
- Audit log format validation
- KSML compliance
"""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit_logger import AuditLogger, get_audit_logger


class TestDay6TTVMetrics(unittest.TestCase):
    """Test TTV intelligence metrics logging"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.test_dir, "logs", "audit")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = AuditLogger(log_dir=self.log_dir, enable_console=False)
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_log_ttv_intelligence_all_metrics(self):
        """Test logging complete TTV intelligence metrics"""
        entry_id = self.logger.log_ttv_intelligence(
            lesson_name="test_lesson",
            story_analysis={
                'character_count': 3,
                'gender_resolved': 3,
                'text_condensation_percent': 15.5,
                'enhanced_prompts_count': 5
            },
            scene_graph={
                'total_scenes': 5,
                'total_entities': 12,
                'avg_entities_per_scene': 2.4,
                'transitions_detected': 4
            },
            narrative={
                'story_beats': 4,
                'character_arcs': 2,
                'avg_tension': 0.65,
                'peak_tension': 0.9,
                'pacing_score': 0.75
            },
            emotion={
                'emotion_changes': 8,
                'avg_motion_intensity': 1.2,
                'emotion_distribution': {'joy': 3, 'fear': 2, 'sadness': 3}
            },
            extension={
                'clips_extended': 3,
                'clips_trimmed': 1,
                'total_clips': 5,
                'avg_extension_duration': 2.5,
                'method': 'combined_slowmo_freeze'
            },
            quality={
                'audio_video_sync_diff': 0.15,
                'total_duration': 45.5,
                'fps': 24,
                'bitrate': '8000k',
                'style': 'realistic'
            },
            ksml_token={
                "ksml_token": "test_token",
                "intent": "video_generation",
                "karma_state": "completed",
                "lineage": {"lesson": "test_lesson", "style": "realistic"}
            }
        )
        
        self.assertIsNotNone(entry_id)
        
        # Read the log file
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        # Validate structure
        self.assertEqual(log_entry['operation'], 'ttv_intelligence_analysis')
        self.assertIn('ttv_metrics', log_entry['metadata'])
        
        metrics = log_entry['metadata']['ttv_metrics']
        self.assertEqual(metrics['story_analysis']['character_count'], 3)
        self.assertEqual(metrics['scene_graph']['total_scenes'], 5)
        self.assertEqual(metrics['narrative']['story_beats'], 4)
        self.assertEqual(metrics['emotion']['emotion_changes'], 8)
        self.assertEqual(metrics['extension']['clips_extended'], 3)
        self.assertAlmostEqual(metrics['quality']['audio_video_sync_diff'], 0.15)
    
    def test_log_ttv_intelligence_partial_metrics(self):
        """Test logging with only some metric categories"""
        entry_id = self.logger.log_ttv_intelligence(
            lesson_name="partial_test",
            story_analysis={'character_count': 2},
            quality={'audio_video_sync_diff': 0.1}
        )
        
        self.assertIsNotNone(entry_id)
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        metrics = log_entry['metadata']['ttv_metrics']
        self.assertEqual(metrics['story_analysis']['character_count'], 2)
        self.assertEqual(metrics['scene_graph'], {})
        self.assertEqual(metrics['narrative'], {})
        self.assertEqual(metrics['emotion'], {})
        self.assertEqual(metrics['extension'], {})
        self.assertAlmostEqual(metrics['quality']['audio_video_sync_diff'], 0.1)
    
    def test_log_performance_metric(self):
        """Test logging module performance metrics"""
        entry_id = self.logger.log_performance_metric(
            module_name="story_parser",
            execution_time=2.5,
            cache_hits=10,
            cache_misses=3
        )
        
        self.assertIsNotNone(entry_id)
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        self.assertEqual(log_entry['operation'], 'module_performance')
        self.assertEqual(log_entry['metadata']['module'], 'story_parser')
        self.assertEqual(log_entry['metadata']['execution_time_seconds'], 2.5)
        self.assertEqual(log_entry['metadata']['cache_hits'], 10)
        self.assertEqual(log_entry['metadata']['cache_misses'], 3)
        
        # Validate cache hit rate calculation
        expected_hit_rate = 10 / (10 + 3)
        self.assertAlmostEqual(log_entry['metadata']['cache_hit_rate'], expected_hit_rate)
    
    def test_log_performance_metric_no_cache(self):
        """Test performance metric with no cache data"""
        entry_id = self.logger.log_performance_metric(
            module_name="test_module",
            execution_time=1.0
        )
        
        self.assertIsNotNone(entry_id)
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        self.assertEqual(log_entry['metadata']['cache_hit_rate'], 0.0)
    
    def test_ksml_compliance(self):
        """Test KSML token compliance in TTV metrics"""
        ksml_token = {
            "ksml_token": "ksml_ttv_test",
            "intent": "video_generation",
            "karma_state": "completed",
            "lineage": {
                "lesson": "test_lesson",
                "style": "realistic",
                "output_path": "/test/path.mp4"
            }
        }
        
        entry_id = self.logger.log_ttv_intelligence(
            lesson_name="ksml_test",
            story_analysis={'character_count': 1},
            ksml_token=ksml_token
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        # Validate KSML compliance structure
        self.assertIn('ksml_compliance', log_entry)
        self.assertEqual(log_entry['ksml_compliance']['token'], 'ksml_ttv_test')
        self.assertEqual(log_entry['ksml_compliance']['intent'], 'video_generation')
        self.assertEqual(log_entry['ksml_compliance']['karma_state'], 'completed')
        self.assertIn('lineage', log_entry['ksml_compliance'])
    
    def test_audit_log_format(self):
        """Test audit log format and structure"""
        self.logger.log_ttv_intelligence(
            lesson_name="format_test",
            story_analysis={'character_count': 2},
            scene_graph={'total_scenes': 3}
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        # Validate required fields
        required_fields = ['entry_id', 'timestamp', 'operation', 'status', 'metadata', 'ksml_compliance']
        for field in required_fields:
            self.assertIn(field, log_entry)
        
        # Validate timestamp format
        timestamp_str = log_entry['timestamp']
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Validate operation
        self.assertEqual(log_entry['operation'], 'ttv_intelligence_analysis')
        self.assertEqual(log_entry['status'], 'success')
    
    def test_multiple_metrics_logging(self):
        """Test logging multiple TTV intelligence entries"""
        lessons = ['lesson1', 'lesson2', 'lesson3']
        
        for lesson in lessons:
            self.logger.log_ttv_intelligence(
                lesson_name=lesson,
                story_analysis={'character_count': len(lesson)},
                quality={'audio_video_sync_diff': 0.1}
            )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)
        
        # Validate each entry
        for i, line in enumerate(lines):
            log_entry = json.loads(line.strip())
            self.assertEqual(log_entry['metadata']['lesson_name'], lessons[i])
    
    def test_emotion_distribution_tracking(self):
        """Test emotion distribution metrics"""
        emotion_dist = {
            'joy': 5,
            'fear': 3,
            'sadness': 2,
            'surprise': 4,
            'neutral': 1
        }
        
        self.logger.log_ttv_intelligence(
            lesson_name="emotion_test",
            emotion={
                'emotion_changes': sum(emotion_dist.values()),
                'avg_motion_intensity': 1.3,
                'emotion_distribution': emotion_dist
            }
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        logged_dist = log_entry['metadata']['ttv_metrics']['emotion']['emotion_distribution']
        self.assertEqual(logged_dist, emotion_dist)
        self.assertEqual(log_entry['metadata']['ttv_metrics']['emotion']['emotion_changes'], 15)
    
    def test_extension_metrics_tracking(self):
        """Test video extension metrics"""
        self.logger.log_ttv_intelligence(
            lesson_name="extension_test",
            extension={
                'clips_extended': 7,
                'clips_trimmed': 2,
                'total_clips': 10,
                'avg_extension_duration': 3.2,
                'method': 'combined_slowmo_freeze'
            }
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        ext_metrics = log_entry['metadata']['ttv_metrics']['extension']
        self.assertEqual(ext_metrics['clips_extended'], 7)
        self.assertEqual(ext_metrics['clips_trimmed'], 2)
        self.assertEqual(ext_metrics['total_clips'], 10)
        self.assertAlmostEqual(ext_metrics['avg_extension_duration'], 3.2)
        self.assertEqual(ext_metrics['method'], 'combined_slowmo_freeze')
    
    def test_narrative_metrics_completeness(self):
        """Test narrative structure metrics"""
        self.logger.log_ttv_intelligence(
            lesson_name="narrative_test",
            narrative={
                'story_beats': 5,
                'character_arcs': 3,
                'avg_tension': 0.68,
                'peak_tension': 0.95,
                'pacing_score': 0.82
            }
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        narr_metrics = log_entry['metadata']['ttv_metrics']['narrative']
        self.assertEqual(narr_metrics['story_beats'], 5)
        self.assertEqual(narr_metrics['character_arcs'], 3)
        self.assertAlmostEqual(narr_metrics['avg_tension'], 0.68)
        self.assertAlmostEqual(narr_metrics['peak_tension'], 0.95)
        self.assertAlmostEqual(narr_metrics['pacing_score'], 0.82)
    
    def test_quality_sync_metrics(self):
        """Test quality and sync metrics"""
        self.logger.log_ttv_intelligence(
            lesson_name="quality_test",
            quality={
                'audio_video_sync_diff': 0.08,
                'total_duration': 62.3,
                'fps': 24,
                'bitrate': '8000k',
                'style': 'cinematic'
            }
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.read().strip())
        
        quality = log_entry['metadata']['ttv_metrics']['quality']
        self.assertAlmostEqual(quality['audio_video_sync_diff'], 0.08)
        self.assertAlmostEqual(quality['total_duration'], 62.3)
        self.assertEqual(quality['fps'], 24)
        self.assertEqual(quality['bitrate'], '8000k')
        self.assertEqual(quality['style'], 'cinematic')
    
    def test_log_file_append_only(self):
        """Test that log file is append-only (immutable)"""
        # Log first entry
        self.logger.log_ttv_intelligence(
            lesson_name="entry1",
            story_analysis={'character_count': 1}
        )
        
        log_file = self.logger.current_log_file
        with open(log_file, 'r') as f:
            first_content = f.read()
        
        # Log second entry
        self.logger.log_ttv_intelligence(
            lesson_name="entry2",
            story_analysis={'character_count': 2}
        )
        
        with open(log_file, 'r') as f:
            second_content = f.read()
        
        # First entry should still be in the file (append-only)
        self.assertIn('entry1', second_content)
        self.assertIn('entry2', second_content)
        self.assertTrue(len(second_content) > len(first_content))
    
    def test_singleton_pattern(self):
        """Test audit logger singleton pattern"""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        
        # Should be the same instance
        self.assertIs(logger1, logger2)


class TestDay6Integration(unittest.TestCase):
    """Test Day 6 integration with production pipeline"""
    
    def test_complete_metrics_workflow(self):
        """Test complete TTV metrics workflow"""
        test_dir = tempfile.mkdtemp()
        log_dir = os.path.join(test_dir, "logs", "audit")
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            logger = AuditLogger(log_dir=log_dir, enable_console=False)
            
            # Simulate complete video generation workflow
            lesson_name = "complete_workflow_test"
            
            # 1. Log TTV intelligence analysis
            logger.log_ttv_intelligence(
                lesson_name=lesson_name,
                story_analysis={
                    'character_count': 4,
                    'gender_resolved': 4,
                    'text_condensation_percent': 12.3,
                    'enhanced_prompts_count': 8
                },
                scene_graph={
                    'total_scenes': 8,
                    'total_entities': 20,
                    'avg_entities_per_scene': 2.5,
                    'transitions_detected': 7
                },
                narrative={
                    'story_beats': 6,
                    'character_arcs': 3,
                    'avg_tension': 0.72,
                    'peak_tension': 0.93,
                    'pacing_score': 0.78
                },
                emotion={
                    'emotion_changes': 12,
                    'avg_motion_intensity': 1.4,
                    'emotion_distribution': {'joy': 4, 'fear': 3, 'sadness': 3, 'surprise': 2}
                },
                extension={
                    'clips_extended': 5,
                    'clips_trimmed': 2,
                    'total_clips': 8,
                    'avg_extension_duration': 2.8,
                    'method': 'combined_slowmo_freeze'
                },
                quality={
                    'audio_video_sync_diff': 0.12,
                    'total_duration': 78.5,
                    'fps': 24,
                    'bitrate': '8000k',
                    'style': 'realistic'
                },
                ksml_token={
                    "ksml_token": "ksml_workflow_test",
                    "intent": "video_generation",
                    "karma_state": "completed",
                    "lineage": {
                        "lesson": lesson_name,
                        "style": "realistic",
                        "output_path": "/test/output.mp4"
                    }
                }
            )
            
            # 2. Log performance metrics
            logger.log_performance_metric(
                module_name="complete_pipeline",
                execution_time=125.5,
                cache_hits=15,
                cache_misses=5
            )
            
            # Validate logs
            log_file = logger.current_log_file
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 2)
            
            # Validate TTV intelligence entry
            ttv_entry = json.loads(lines[0].strip())
            self.assertEqual(ttv_entry['operation'], 'ttv_intelligence_analysis')
            self.assertEqual(ttv_entry['metadata']['lesson_name'], lesson_name)
            
            # Validate performance entry
            perf_entry = json.loads(lines[1].strip())
            self.assertEqual(perf_entry['operation'], 'module_performance')
            self.assertEqual(perf_entry['metadata']['module'], 'complete_pipeline')
            
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
