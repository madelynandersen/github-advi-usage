# Classification Rules

## Query-Result Inclusion

`why_in_query_results` uses the following labels:

- `true_positive`: the row is a user-level VI fit call included in analysis.
- `import_only`: the search hit only reflected an import or symbol reference.
- `comment_only`: the search hit only reflected a comment or inactive example.
- `no_vi_fit`: the search hit did not represent a user-level VI fitting call.
- `method_repo_code_itself`: the search hit came from PyMC, NumPyro, TFP, or
  Stan package source/tests/internals rather than an external application repo.
- `needs_review`: the row still needs manual review and is excluded from the
  main aggregate summary for now.

Rows where `why_in_query_results` is not `true_positive` have `NA` for
`call_kind`, `actual_call_snippet`, `extracted_function`,
`extracted_arguments`, and `classification_reason`.

## Optimizer and MC Adjustments (All Packages)

- An optimizer adjustment means using a different optimizer than the
  package's default (e.g. SGD or AdaGrad instead of Adam). Changing only the
  learning rate (or other hyperparameters) of the default optimizer does not
  count as an adjustment.
- A call site that adjusts both the optimizer and the MC/sample count is
  classified `adjusted_both`; every adjustment present is recorded.

## PyMC

Relevant parameters:

- `obj_optimizer`
- `obj_n_mc`

Ignored parameters:

- `n`
- `progressbar`
- `callbacks`
- `random_seed`

Default behavior:

- no relevant optimizer argument
- no relevant MC argument
- `obj_n_mc=None`, which falls back to PyMC default behavior
- `adagrad_window` where it is PyMC's default optimizer behavior

Adjusted behavior:

- explicit non-default `obj_optimizer`
- non-default `obj_n_mc`

## TensorFlow Probability

Relevant parameters:

- `optimizer`
- `sample_size`

Ignored parameters:

- `surrogate_posterior`
- `num_steps`
- `target_log_prob_fn`

Default behavior:

- Adam optimizer (at any learning rate)
- no relevant sample-size change

Adjusted behavior:

- any optimizer that is not Adam
- changed `sample_size`

## Pyro and NumPyro

Relevant parameters:

- optimizer choice in `SVI(...)`
- ELBO/loss setup, especially the Monte Carlo sample count via
  `num_particles`

Default behavior:

- Adam or `ClippedAdam` optimizer (at any learning rate)
- `Trace_ELBO()` or `Trace_MeanField_ELBO()` with no non-default parameters
- `num_particles=1`

`ClippedAdam` counts as default alongside Adam because Pyro's own guidance
recommends both as the default SVI optimizer: "Use Adam or ClippedAdam by
default when doing Stochastic Variational Inference. Note that `ClippedAdam`
is just a convenient extension of `Adam` that provides built-in support for
learning rate decay and gradient clipping." ([SVI Part IV: Tips and Tricks,
section 2](https://pyro.ai/examples/svi_part_iv.html))

Adjusted behavior:

- any optimizer that is not Adam or `ClippedAdam`, including `AdamW`,
  `DCTAdam`, scheduled optimizers, or wrappers around non-default optimizers
- `num_particles > 1`
- custom/non-default ELBO/loss functions

For this study, Pyro/NumPyro `num_particles` is treated as the same conceptual
quantity as MC/sample count in the other packages.

## Stan

Covered interfaces: CmdStanPy `model.variational(...)` and PyStan 2
`model.vb(...)` (ADVI). PyStan 3 has no VI method. CmdStanPy's
`model.pathfinder(...)` is a different VI algorithm and is out of scope.

Relevant parameter:

- `grad_samples` (MC draws per gradient step; default 1)

Stan's ADVI has no optimizer choice -- it always uses its own adaptive
step-size sequence -- so a Stan call site can only be `default` or
`adjusted_mc`. `eta` (the step size) is a learning rate and is ignored, as are
`elbo_samples`, `tol_rel_obj`, `adapt_engaged`, `iter`, and `algorithm`
(`meanfield` vs. `fullrank` is the approximation family, not an optimizer or
MC setting).

Default behavior:

- `grad_samples` absent or `1`

Adjusted behavior:

- `grad_samples` set to a number other than 1

A `grad_samples` value that can't be resolved to a number (e.g. a config
lookup) is routed to `needs_review`.

Where available, the Stan variational algorithm is recorded as `meanfield`,
`fullrank`, or configurable.

