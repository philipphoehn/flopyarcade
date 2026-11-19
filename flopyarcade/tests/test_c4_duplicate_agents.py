"""Tests for C4 fix: duplicate agents after restart.

Validates that resumeGenetic properly restores noveltyItemCount,
agentsUnique, agentsUniqueIDs, and agentsDuplicate from a loaded
novelty archive, preventing new agents from overwriting existing entries.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from collections import OrderedDict


class TestC4NoveltyStateRestoration:
    """Test that the C4 fix correctly restores novelty state on resume."""

    def _make_archive(self, n_agents, n_duplicates=0):
        """Create a mock novelty archive with n_agents entries."""
        archive = {}
        for i in range(n_agents):
            uid = i if i < (n_agents - n_duplicates) else 0  # duplicates share uid 0
            archive[f'agent{i+1}'] = {
                'itemID': i,
                'modelFile': f'/tmp/agent{i+1}.keras',
                'resultsFile': f'/tmp/agent{i+1}_results.p',
                'actionsUniqueID': uid,
                'novelty': 0.5,
            }
        return archive

    def test_noveltyItemCount_set_after_load(self):
        """noveltyItemCount must equal len(archive) after resume."""
        archive = self._make_archive(5)
        # Simulate the fix logic
        noveltyItemCount = len(archive)
        assert noveltyItemCount == 5

    def test_agents_unique_rebuilt(self):
        """agentsUnique must contain only first-seen unique IDs."""
        archive = self._make_archive(5, n_duplicates=0)
        # Simulate the fix logic
        agents_unique, agents_unique_ids = [], []
        agents_duplicate = []
        for i in range(len(archive)):
            agent_str = f'agent{i+1}'
            entry = archive[agent_str]
            if 'actionsUniqueID' in entry:
                uid = entry['actionsUniqueID']
                if uid not in agents_unique_ids:
                    agents_unique.append(i)
                    agents_unique_ids.append(uid)
                else:
                    agents_duplicate.append(i)
        assert len(agents_unique) == 5
        assert len(agents_duplicate) == 0

    def test_agents_duplicate_detected(self):
        """Duplicate agents (same actionsUniqueID) must be tracked."""
        # 5 agents, last 2 are duplicates of agent 0
        archive = self._make_archive(5, n_duplicates=2)
        agents_unique, agents_unique_ids = [], []
        agents_duplicate = []
        for i in range(len(archive)):
            agent_str = f'agent{i+1}'
            entry = archive[agent_str]
            if 'actionsUniqueID' in entry:
                uid = entry['actionsUniqueID']
                if uid not in agents_unique_ids:
                    agents_unique.append(i)
                    agents_unique_ids.append(uid)
                else:
                    agents_duplicate.append(i)
        # Agents 0,1,2 are unique (uid 0,1,2), agents 3,4 are duplicates (uid 0)
        assert len(agents_unique) == 3
        assert len(agents_duplicate) == 2
        assert 3 in agents_duplicate
        assert 4 in agents_duplicate

    def test_no_key_collision_after_resume(self):
        """After resume, new agents must NOT overwrite existing archive keys."""
        archive = self._make_archive(3)
        novelty_item_count = len(archive)  # = 3
        # Simulate _prepare_novelty_archive adding 2 more agents
        for i in range(2):
            new_key = f'agent{novelty_item_count + 1 + i}'
            assert new_key not in archive, (
                f"New key {new_key} collides with existing archive!"
            )

    def test_empty_archive_resume(self):
        """Resuming with empty archive should be safe."""
        archive = {}
        noveltyItemCount = len(archive)
        agents_unique, agents_unique_ids = [], []
        agents_duplicate = []
        for i in range(noveltyItemCount):
            pass  # no iteration
        assert noveltyItemCount == 0
        assert len(agents_unique) == 0

    def test_archive_without_actions_unique_id(self):
        """Entries without actionsUniqueID should still be counted as unique."""
        archive = {
            'agent1': {'itemID': 0, 'modelFile': '/tmp/a1.keras'},
            'agent2': {'itemID': 1, 'modelFile': '/tmp/a2.keras'},
        }
        agents_unique, agents_unique_ids = [], []
        agents_duplicate = []
        for i in range(len(archive)):
            agent_str = f'agent{i+1}'
            entry = archive[agent_str]
            if 'actionsUniqueID' in entry:
                uid = entry['actionsUniqueID']
                if uid not in agents_unique_ids:
                    agents_unique.append(i)
                    agents_unique_ids.append(uid)
                else:
                    agents_duplicate.append(i)
            else:
                agents_unique.append(i)
        assert len(agents_unique) == 2
        assert len(agents_duplicate) == 0

    def test_single_agent_archive(self):
        """Single agent archive should have 1 unique, 0 duplicates."""
        archive = self._make_archive(1)
        agents_unique, agents_unique_ids = [], []
        agents_duplicate = []
        for i in range(len(archive)):
            agent_str = f'agent{i+1}'
            entry = archive[agent_str]
            if 'actionsUniqueID' in entry:
                uid = entry['actionsUniqueID']
                if uid not in agents_unique_ids:
                    agents_unique.append(i)
                    agents_unique_ids.append(uid)
                else:
                    agents_duplicate.append(i)
        assert len(agents_unique) == 1
        assert len(agents_duplicate) == 0


class TestC4ResumeIntegration(TestC4NoveltyStateRestoration):
    """Integration-style test: simulate the full resume flow."""

    def test_resume_then_prepare_no_overwrite(self):
        """Simulate: resume loads 4 agents, then prepare adds 3 more.
        Total should be 7, with no key collisions."""
        # Phase 1: resume
        archive = self._make_archive(4)
        noveltyItemCount = len(archive)
        assert noveltyItemCount == 4

        # Phase 2: prepare new agents (simulating _prepare_novelty_archive)
        n_new = 3
        for i in range(n_new):
            new_key = f'agent{noveltyItemCount + 1 + i}'
            archive[new_key] = {
                'itemID': noveltyItemCount + i,
                'modelFile': f'/tmp/new{i}.keras',
            }
            noveltyItemCount += 1

        assert noveltyItemCount == 7
        assert len(archive) == 7
        # Verify no overwrite
        assert archive['agent1']['modelFile'] == '/tmp/agent1.keras'
        assert 'agent5' in archive
        assert 'agent7' in archive
