
Supported Algorithms
======================

Following algorithms are supported:

* Multi-task SAC

* Multi-task SAC with Task Encoder

* Multi-headed SAC

* Distral from `Distral: Robust multitask reinforcement learning` :cite:`2017distral`

* PCGrad from `Gradient surgery for multi-task learning` :cite:`2020pcgrad`

* GradNorm from `Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks` :cite:`2018gradnorm`

* DeepMDP from `DeepMDP: Learning Continuous Latent Space Models for Representation Learning` :cite:`2019deepmdp`

* HiPBMDP from `Multi-Task Reinforcement Learning as a Hidden-Parameter Block MDP` :cite:`2020mtrl_as_a_hidden_block_mdp`

* Soft Modularization from `Multi-Task Reinforcement Learning with Soft Modularization` :cite:`2020soft_modularization`

* CARE

Along with the standard SAC components (actor, critic, etc), following components are supported and can be used with the base algorithms in plug-and-play fashion:

* Task Encoder

* State Encoders

    * Attention weighted Mixture of Encoders
    * Gated Mixture of Encoders
    * Ensemble of Encoders
    * FiLM Encoders :cite:`2018film`

* Multi-headed actor, critic, value-fuctions

* Modularized actor, critic, value-fuctions based on :cite:`2020soft_modularization`

For example, we can train a Multi-task SAC with FiLM encoders or train Multi-headed SAC with gated mixture of encoders with or without using task encoders.

Refer to the `tutorial <>`_ for more details.

References
-------------

.. bibliography::