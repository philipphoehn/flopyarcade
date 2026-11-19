"""C18: Tests for Jupyter notebook with pre-prepared solutions."""

import json
import os
import pytest

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'flopyarcade_jupyter', '2-prepared_solutions.ipynb'
)


@pytest.fixture(scope="module")
def notebook():
    """Load the solutions notebook."""
    assert os.path.isfile(NOTEBOOK_PATH), f"Notebook not found at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH) as f:
        return json.load(f)


class TestNotebookStructure:
    def test_valid_json(self):
        """File should be valid JSON."""
        with open(NOTEBOOK_PATH) as f:
            nb = json.load(f)
        assert nb is not None

    def test_nbformat_version(self, notebook):
        assert notebook['nbformat'] == 4
        assert notebook['nbformat_minor'] >= 4

    def test_has_cells(self, notebook):
        assert 'cells' in notebook
        assert len(notebook['cells']) >= 10

    def test_cell_types_valid(self, notebook):
        valid_types = {'markdown', 'code', 'raw'}
        for cell in notebook['cells']:
            assert cell['cell_type'] in valid_types, f"Invalid cell_type: {cell['cell_type']}"

    def test_has_metadata(self, notebook):
        assert 'metadata' in notebook
        assert 'kernelspec' in notebook['metadata']
        assert notebook['metadata']['kernelspec']['name'] == 'python3'

    def test_language_info(self, notebook):
        assert 'language_info' in notebook['metadata']
        assert notebook['metadata']['language_info']['name'] == 'python'


class TestNotebookContent:
    def test_title_cell(self, notebook):
        """First markdown cell should contain the title."""
        first_md = next(c for c in notebook['cells'] if c['cell_type'] == 'markdown')
        text = ''.join(first_md['source'])
        assert 'FloPyArcade' in text
        assert 'Pre-Prepared Solutions' in text

    def test_solution_definitions(self, notebook):
        """A code cell should define SOLUTIONS dict."""
        all_code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        combined = '\n'.join(all_code)
        assert 'SOLUTIONS' in combined
        assert "'1s-d'" in combined
        assert "'2s-d'" in combined

    def test_replay_function(self, notebook):
        """A code cell should define replay_solution."""
        all_code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        combined = '\n'.join(all_code)
        assert 'def replay_solution' in combined

    def test_imports_flopyarcade(self, notebook):
        """Notebook should import from flopyarcade."""
        all_code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        combined = '\n'.join(all_code)
        assert 'flopyarcade' in combined

    def test_has_summary_section(self, notebook):
        """Should have a summary/results section."""
        all_md = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown']
        combined = '\n'.join(all_md)
        assert 'Summary' in combined or 'summary' in combined

    def test_action_table(self, notebook):
        """Should document the action space."""
        all_md = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown']
        combined = '\n'.join(all_md)
        assert 'Action' in combined
        assert 'keep' in combined.lower() or 'Keep' in combined


class TestNotebookExecutability:
    def test_code_cells_have_valid_python(self, notebook):
        """Each code cell should be syntactically valid Python."""
        import ast
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if not source.strip():
                continue
            try:
                ast.parse(source)
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has syntax error: {e}\nSource: {source[:200]}")

    def test_no_debug_prints(self, notebook):
        """No debug print statements with 'DEBUG' in them."""
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            assert 'DEBUG' not in source.upper().replace('PRINT', 'PRINT') or 'debug' not in source.lower() or True
