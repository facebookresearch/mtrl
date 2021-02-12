MTRL
====

Multi Task RL Algorithms

Introduction
------------

MTRL is a library of multi-task reinforcement learning algorithms. It has two main components:


* `Components <https://github.com/facebookresearch/mtrl/tree/master/mtrl/agent/components>`_ and `agents <https://github.com/facebookresearch/mtrl/tree/master/mtrl/agent>`_ that implement the multi-task RL algorithms.

* `Experiment setups <https://github.com/facebookresearch/mtrl/tree/master/mtrl/experiment>`_ that enable training/evaluation on different setups. 

Together, these two components enable use of MTRL across different environments and setups.

List of publications & submissions using MTRL (please create a pull request to add the missing entries):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Learning Robust State Abstractions for Hidden-Parameter Block MDPs <https://arxiv.org/abs/2007.07206>`_

License
^^^^^^^

* MTRL uses `MIT License <https://github.com/facebookresearch/mtrl/blob/main/LICENSE>`_.

* `Terms of Use <https://opensource.facebook.com/legal/terms>`_

* `Privacy Policy <https://opensource.facebook.com/legal/privacy>`_

Citing MTRL
^^^^^^^^^^^

If you use MTRL in your research, please use the following BibTeX entry:

.. code-block::

   @Misc{Sodhani2021MTRL,
     author =       {Shagun Sodhani, Amy Zhang},
     title =        {MTRL - Multi Task RL Algorithms},
     howpublished = {Github},
     year =         {2021},
     url =          {https://github.com/facebookresearch/mtrl}
   }

Setup
-----

* Clone the repository: ``git clone git@github.com:facebookresearch/mtrl.git``.

* Install dependencies: ``pip install -r requirements/dev.txt``

Usage
-----

* MTRL supports many different multi-task RL algorithms as described `here <https://mtrl.readthedocs.io/en/latest/pages/algorithms/supported.html>`_.

* MTRL supports multi-task environments using `MTEnv <https://github.com/facebookresearch/mtenv>`_. These environments include `MetaWorld <https://meta-world.github.io/>`_ and multi-task variants of `DMControl Suite <https://github.com/deepmind/dm_control>`_

* Refer the `tutorial]<https://mtrl.readthedocs.io/en/latest/pages/tutorials/overview.html>`_ to get started with MTRL.

Documentation
-------------

`https://mtrl.readthedocs.io <https://mtrl.readthedocs.io>`_

Contributing to MTRL
--------------------

There are several ways to contribute to MTRL.


#. Use MTRL in your research.

#. Contribute a new algorithm. The currently supported algorithms are listed `here <https://mtrl.readthedocs.io/en/latest/pages/algorithms/supported.html>`_ and are looking forward to adding more algorithms.

#. Check out the `beginner-friendly <https://github.com/facebookresearch/mtrl/pulls?q=is%3Apr+is%3Aopen+label%3A%22good+first+issue%22>`_ issues on GitHub and contribute to fixing those issues.

#. Check out additional details `here <https://github.com/facebookresearch/mtrl/blob/main/.github/CONTRIBUTING.md>`_.

Community
---------

Ask questions in the chat or github issues:

* `Chat <https://mtenv.zulipchat.com>`_
* `Issues <https://https://github.com/facebookresearch/mtrl/issues>`_
