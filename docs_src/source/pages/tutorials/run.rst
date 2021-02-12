
Running Code
============

The code uses `Hydra <https://github.com/facebookresearch/hydra>`_ framework 
for composing configs and running the code. Let us say that we want to 
train multi-task SAC on MT10 (from MetaWorld). The command for that will
look like:

.. code-block:: bash

    PYTHONPATH=. python3 -u main.py \
    setup=metaworld \
    agent=state_sac \
    env=metaworld-mt10 \
    agent.multitask.num_envs=10 \
    agent.multitask.should_use_disentangled_alpha=True

Let us break this command piece by piece.

* 
  ``setup=metaworld`` says that we want to use a specific setup called ``metaworld``. 
  In multitask RL, different works/environments care about different setups. For 
  example, commonly multitask RL environments use episodic rewards as the metric 
  to optimize while MetaWorld  :cite:`2020metaworld` use ``success`` 
  as the key metric. Some multitask RL setups evaluate the agent on the same 
  set of environments that it was trained on while HiPBMDP :cite:`2020mtrl_as_a_hidden_block_mdp` 
  evalutes on three sets of unseen environments. We abstract away these details via 
  ``setup`` parameter. Supported values are listed `here <https://github.com/facebookresearch/mtrl/tree/master/config/setup>`_. 
  When a setup is selected, the corresponding `metrics config <https://github.com/facebookresearch/mtrl/tree/master/config/metrics>`_ is also loaded. By default, we also load optimizers and agent 
  components based on the ``setup`` value but this can be easily overided 
  (as described in the next step). We can easily add a new setup, by defining 
  a new config or updating the existing configs. For example, to add the 
  ``hipbdmp`` setup, we added a `metrics config <https://github.com/facebookresearch/mtrl/tree/master/config/metrics>`_ 
  and new `optimizer configs <https://github.com/facebookresearch/mtrl/tree/master/config/agent/optimizers>`_ 
  assuming the values should be different for the new setup. But we do not 
  change the `agent configs <https://github.com/facebookresearch/mtrl/tree/master/config/agent>`_
  as the agent implementation does not have to change with the setup, though 
  the user is free to update the agent configs as well.

* 
  ``agent=state_sac`` says that we want to train SAC using state observations.


  * Other supported agents are listed as top-level yaml files `here <https://github.com/facebookresearch/mtrl/tree/master/config/agent>`_.
  * Update the config files to change the agent's hyper-parameters.
  * Add a new config file to support a new agent.
  * You would note that we are using the ``setup`` value in the name of 
    `component configs <https://github.com/facebookresearch/mtrl/tree/master/config/agent/components>`_ 
    and `optimizer configs <https://github.com/facebookresearch/mtrl/tree/master/config/agent/optimizers>`_. 
    This is completely optional and the same effect can be achieved by 
    command line overrides. We opt for using multiple config to reduce the 
    overhead of remembering what values to override when running the code.

* 
  ``env=metaworld-mt10`` says that we want to train on MT10 environment from MetaWorld.


  * Other supported environments are listed `here <https://github.com/facebookresearch/mtrl/tree/master/config/env>`_.
  * New environments can be added by creating a new config file in the directory above.

* 
  ``agent.multitask.num_envs=10`` sets the number of tasks to be 10.

* ``agent.multitask.should_use_disentangled_alpha=True`` says that we want to 
  learn a different entropy coefficient for each task.

We can update the previous command to train multi-task multi-headed SAC 
agent by adding an additional argument ``agent.multitask.should_use_multi_head_policy=True`` 
as follows:

.. code-block:: bash

   PYTHONPATH=. python3 -u main.py \
   setup=metaworld \
   agent=state_sac \
   env=metaworld-mt10 \
   agent.multitask.num_envs=10 \
   agent.multitask.should_use_disentangled_alpha=True \
   agent.multitask.should_use_multi_head_policy=True 

We can control more aspetts of training (like seed, number of training 
steps, batch size etc) by adding ad the previous command to train 
multi-task multi-headed SAC agent by adding additional arguments as follows:

.. code-block:: bash

   PYTHONPATH=. python3 -u main.py \
   setup=metaworld \
   env=metaworld-mt10 \
   agent=state_sac \
   agent.multitask.num_envs=10 \
   agent.multitask.should_use_disentangled_alpha=True \
   agent.multitask.should_use_multi_head_policy=True \
   experiment.num_train_steps=2000000 \
   setup.seed=1 \
   replay_buffer.batch_size=1280

* 
  ``experiment.num_train_steps=2000000`` says that we should train the agent
  for 2 million steps.

* 
  ``setup.seed=1`` sets the seed to 1.

* 
  ``replay_buffer.batch_size=1280`` says that batches, sampled from replay 
  buffer, will contain 1280 transitions.


  