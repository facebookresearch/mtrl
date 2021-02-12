
Baselines
============

DMControl
^^^^^^^^^

Distral
~~~~~~~

.. code-block:: bash

    MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=. python3 -u main.py \
    setup=hipbmdp \
    env=dmcontrol-finger-spin-distribution-v0 \
    agent=distral \
    setup.seed=1 \
    agent.distral_alpha=1.0 \
    agent.distral_beta=1.0 \
    replay_buffer.batch_size=256 

DeepMDP
~~~~~~~

.. code-block:: bash

    MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=. python3 -u main.py \
    setup=hipbmdp \
    env=dmcontrol-finger-spin-distribution-v0 \
    agent=deepmdp \
    setup.seed=1 \
    replay_buffer.batch_size=256

GradNorm
~~~~~~~~

.. code-block:: bash

    MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=. python3 -u main.py \
    setup=hipbmdp \
    env=dmcontrol-finger-spin-distribution-v0 \
    agent=deepmdp \
    setup.seed=1 \
    replay_buffer.batch_size=256

PCGrad
~~~~~~

.. code-block:: bash

    MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=. python3 -u main.py \
    setup=hipbmdp \
    env=dmcontrol-finger-spin-distribution-v0 \
    agent=pcgrad_sac \
    setup.seed=1 \
    replay_buffer.batch_size=256 

HiP-BMDP
~~~~~~~~

.. code-block:: bash

    MUJOCO_GL="osmesa" LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH PYTHONPATH=. python3 -u main.py \
    setup=hipbmdp \
    env=dmcontrol-finger-spin-distribution-v0 \
    agent=hipbmdp \
    setup.seed=1 \
    replay_buffer.batch_size=256

Metaworld
^^^^^^^^^

Multi-task SAC
~~~~~~~~~~~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True \
    agent.encoder.type_to_select=identity \
    agent.multitask.should_use_multi_head_policy=False \
    agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
    agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True \
    agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True

Multi-task Multi-headed SAC
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True \
    agent.encoder.type_to_select=identity \
    agent.multitask.should_use_multi_head_policy=True \
    agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
    agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
    agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False 

PCGrad
~~~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=pcgrad_state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=False \
    agent.multitask.should_use_task_encoder=False \
    agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
    agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
    agent.encoder.type_to_select=identity 

SoftModularization
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True \
    agent.multitask.should_use_task_encoder=True \
    agent.encoder.type_to_select=feedforward \
    agent.multitask.actor_cfg.should_condition_model_on_task_info=True \
    agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
    agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
    agent.multitask.actor_cfg.moe_cfg.should_use=True \
    agent.multitask.actor_cfg.moe_cfg.mode=soft_modularization \
    agent.multitask.should_use_multi_head_policy=False \
    agent.encoder.feedforward.hidden_dim=50 \
    agent.encoder.feedforward.num_layers=2 \
    agent.encoder.feedforward.feature_dim=50 \
    agent.actor.num_layers=4 \
    agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=False 

SAC + FiLM Encoder
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True \
    agent.multitask.should_use_task_encoder=True \
    agent.encoder.type_to_select=film \
    agent.multitask.should_use_multi_head_policy=False \
    agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=True \
    agent.multitask.task_encoder_cfg.model_cfg.output_dim=6 

CARE
~~~~

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    env=metaworld-mt10 \
    agent=state_sac \
    experiment.num_eval_episodes=1 \
    experiment.num_train_steps=2000000 \
    setup.seed=1 \
    replay_buffer.batch_size=1280 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True \
    agent.multitask.should_use_task_encoder=True \
    agent.encoder.type_to_select=moe \
    agent.multitask.should_use_multi_head_policy=False \
    agent.encoder.moe.task_id_to_encoder_id_cfg.mode=attention \
    agent.encoder.moe.num_experts=4 \
    agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=True
