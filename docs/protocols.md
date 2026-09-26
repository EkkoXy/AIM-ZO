# Evaluation protocol

A run is defined by its model revision, task, split construction, prompt and
verbalizer, train/dev sample counts, seed, dtype, method parameters, forward
budget, and checkpoint-selection rule.

The main protocol uses seeds 42, 142, and 242. Training data is split into a
1,000-example train subset and a disjoint 500-example development subset when
the task provides enough examples. Five uniformly spaced checkpoints are
evaluated on this train-derived development set. The selected checkpoint is
then evaluated once on the official validation split.

Classification tasks report accuracy. MultiRC additionally reports grouped F1a
and exact match. SQuAD and DROP report generated-answer F1 and exact match.

Forward budgets count every objective evaluation. AIM-ZO uses one center plus
fifteen probes per optimization step. Configurations retain the exact step and
forward-call budgets used for each experiment.
