FloPyArcade Documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Introduction
------------

FloPyArcade is a reinforcement-learning framework built on the
MODFLOW 6 / MODPATH 6 groundwater simulation suite. It exposes
Gym-style environments where an agent places extraction and
injection wells to maximise a user-defined reward.

Key concepts
~~~~~~~~~~~~

- **Environment** – a ``gym.Env`` subclass wrapping a MODFLOW grid.
- **Agent** – a policy (random, human, or neural) that selects actions.
- **Reward** – a scalar score computed after each episode.
- **Renderer** – draws the 2-D / 3-D head / trajectory visualisation.

Quick start
~~~~~~~~~~~

.. code-block:: bash

   pip install flopyarcade
   python -m flopyarcade.play --envtype 1s-d --ngames 5

API Reference
-------------

See :doc:`api` for detailed module documentation.
