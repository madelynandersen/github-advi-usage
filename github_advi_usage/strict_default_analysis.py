"""Secondary, stricter default-vs-non-default analysis of true-positive VI call sites.

The main pipeline (`analyze`) only asks whether the optimizer and the
MC/sample count were left at their defaults. This analysis asks whether *any*
input that can change the fitted approximation was moved off the value the
package would use automatically. Number of iterations is ignored for
Pyro/NumPyro/TFP but not for PyMC or Stan. The rules, the defaults checked, and
every parameter deliberately ignored (with the reason) are defined in the
tables below and written alongside the results.

Reads the database read-only and writes only to its own output folder:

    python -m github_advi_usage.strict_default_analysis \
        --db data/github-vi-software-stan.sqlite \
        --out data/github-vi-software-stan/strict_default_reports

A call site is classified as one of:
  default      every checked input is at its automatic value
  non_default  at least one checked input is set to something else
  unresolved   no checked input is non-default, but at least one value can't be
               read statically (e.g. `Adam({"lr": lr})` where `lr` is a
               function argument)

Each call site is judged on its whole configuration group: the setup call
the pipeline links it to (its "settings call") plus every other true-positive
call linked to that same settings call (guide constructors, `svi.run(...)`,
`pm.ADVI(...)`, surrogate-posterior builders), so e.g. an `svi.step()` inherits
an `init_loc_fn` set on its guide.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_vi_github_analysis import call_site_id, final_from_relevant
from .github_advi import GITHUB_ADVI_RELEVANT_PARAMETER_ANALYSIS_QUERY, query_rows

# The secondary analysis covers NumPyro, PyMC, TFP and Stan. Pyro is excluded,
# and so are call sites whose imports don't show whether they use Pyro or
# NumPyro (package "pyro_or_numpyro").
EXCLUDED_PACKAGES = {"pyro", "pyro_or_numpyro"}
PACKAGE_GROUPS = {
    "pymc": "PyMC",
    "tensorflow_probability": "TensorFlow Probability",
    "numpyro": "NumPyro",
    "stan": "Stan",
}
GROUP_ORDER = ["PyMC", "TensorFlow Probability", "NumPyro", "Stan"]


def without_pyro(table: list[tuple]) -> list[tuple]:
    """Rule-table rows for the packages in this analysis (Pyro rows dropped,
    shared Pyro/NumPyro rows relabeled NumPyro)."""
    out = []
    for package, *rest in table:
        if package == "pyro":
            continue
        out.append((package.replace("pyro / numpyro", "numpyro"), *rest))
    return out

# --------------------------------------------------------------------------
# Rule tables. CHECKED_PARAMETERS and IGNORED_PARAMETERS are the documentation
# of record: they're written verbatim to the output folder.
# --------------------------------------------------------------------------

CHECKED_PARAMETERS: list[tuple[str, str, str, str]] = [
    # (package, parameter, automatic/default value, rule)
    ("pyro", "optimizer", "pyro.optim.Adam or pyro.optim.ClippedAdam", "Any other optimizer is non-default."),
    ("pyro", "optimizer lr", "1e-3", "Adam/ClippedAdam are default only with lr = 1e-3 (also the value used if 'lr' is omitted)."),
    ("pyro", "optimizer betas", "(0.9, 0.999)", "Default only if omitted or equal to (0.9, 0.999)."),
    ("pyro", "optimizer eps / weight_decay / amsgrad", "1e-8 / 0 / False", "Any other value is non-default."),
    ("pyro", "ClippedAdam clip_norm / lrd", "10.0 / 1.0", "Any other value is non-default."),
    ("pyro", "optimizer clip_args", "None", "Gradient clipping wrapper; any value is non-default."),
    ("pyro", "loss", "Trace_ELBO or TraceMeanField_ELBO (incl. JIT variants)", "Any other ELBO (TraceGraph_ELBO, TraceEnum_ELBO, RenyiELBO, custom) is non-default."),
    ("pyro", "loss num_particles", "1", "Any other value is non-default; writing num_particles=1 explicitly is default."),
    ("pyro", "SVI loss_and_grads", "None", "Custom gradient function; any value is non-default."),
    ("pyro", "guide init_loc_fn", "init_to_feasible (AutoNormal), init_to_median (other autoguides)", "Changes initial parameter values; anything other than the guide's own default is non-default."),
    ("pyro", "guide init_scale", "0.1", "Changes initial scale; any other value is non-default."),
    ("numpyro", "optimizer", "numpyro.optim.Adam", "Any other optimizer (ClippedAdam, optax, Adagrad, ...) is non-default."),
    ("numpyro", "optimizer step_size", "5e-4", "Adam is default only with step_size = 5e-4; a schedule is non-default."),
    ("numpyro", "optimizer b1 / b2 / eps", "0.9 / 0.999 / 1e-8", "Any other value is non-default."),
    ("numpyro", "loss", "Trace_ELBO or TraceMeanField_ELBO", "Any other ELBO is non-default."),
    ("numpyro", "loss num_particles", "1", "Any other value is non-default."),
    ("numpyro", "loss multi_sample_guide", "False", "Changes the ELBO estimator; True is non-default."),
    ("numpyro", "guide init_loc_fn / init_strategy", "init_to_uniform", "Changes initial parameter values; anything else is non-default."),
    ("numpyro", "guide init_scale", "0.1", "Changes initial scale; any other value is non-default."),
    ("numpyro", "svi.run init_params / init_state", "None", "Starts the optimization from supplied values; any value is non-default."),
    ("tensorflow_probability", "optimizer", "Keras Adam with no step size (tf.optimizers.Adam(), tf.keras/tf_keras.optimizers.Adam(), tf.compat.v1.train.AdamOptimizer())", "Any other optimizer is non-default, including optax.adam (JAX substrate), which has no automatic step size."),
    ("tensorflow_probability", "optimizer learning_rate", "not given (Keras default 0.001)", "Any explicit value other than 0.001 is non-default; an explicit 0.001 equals the automatic value and is counted as default. A schedule is non-default."),
    ("tensorflow_probability", "optimizer beta_1 / beta_2 / epsilon / amsgrad", "0.9 / 0.999 / 1e-7 / False", "Any other value is non-default."),
    ("tensorflow_probability", "optimizer clipnorm / clipvalue / global_clipnorm / weight_decay / use_ema", "None / False", "Any value is non-default."),
    ("tensorflow_probability", "sample_size", "1", "Any other value is non-default."),
    ("tensorflow_probability", "importance_sample_size", "1", "Any other value is non-default."),
    ("tensorflow_probability", "variational_loss_fn / discrepancy_fn", "kl_reverse", "Any other divergence is non-default."),
    ("tensorflow_probability", "surrogate builder initial_* arguments", "not given", "Changes initial values (e.g. initial_unconstrained_loc); any value is non-default."),
    ("pymc", "n (iterations)", "10000", "Any other value is non-default."),
    ("pymc", "obj_n_mc", "1", "Any other value is non-default."),
    ("pymc", "obj_optimizer", "adagrad_window(learning_rate=0.001, epsilon=0.1, n_win=10)", "Any other optimizer, or adagrad_window with any other setting, is non-default."),
    ("pymc", "start / start_sigma (on fit, ADVI(...), or inside inf_kwargs)", "None", "Changes initial values; any value is non-default."),
    ("pymc", "callbacks containing CheckParametersConvergence", "none", "Stops before n iterations, so the iteration count is no longer 10000; non-default."),
    ("pymc", "total_grad_norm_constraint / more_obj_params / more_updates", "None", "Changes the gradient or the parameters being optimized; any value is non-default."),
    ("pymc", "method", "'advi' / 'fullrank_advi' / ADVI() / FullRankADVI()", "Family choice is ignored (guide choice doesn't matter); a non-ADVI method (e.g. 'svgd') is non-default."),
    ("stan", "iter", "10000", "Any other value is non-default."),
    ("stan", "grad_samples", "1", "Any other value is non-default."),
    ("stan", "elbo_samples", "100", "Any other value is non-default."),
    ("stan", "eta", "1.0 (cmdstanpy None means CmdStan's 1.0)", "Any other value is non-default."),
    ("stan", "adapt_engaged", "True", "Any other value is non-default."),
    ("stan", "adapt_iter", "50", "Any other value is non-default."),
    ("stan", "tol_rel_obj", "0.01", "Changes when VB declares convergence; any other value is non-default."),
    ("stan", "eval_elbo", "100", "Changes how often convergence is checked; any other value is non-default."),
    ("stan", "inits / init", "None (cmdstanpy) / 'random' (PyStan 2)", "Changes initial values; any other value is non-default."),
]

IGNORED_PARAMETERS: list[tuple[str, str, str]] = [
    # (package, parameter, why it's ignored)
    ("pyro / numpyro / tfp", "number of iterations (num_steps, loop length, svi.run num_steps)", "Per the analysis definition, iteration count is not a default/non-default criterion for Pyro, NumPyro, or TFP."),
    ("tensorflow_probability", "convergence_criterion", "A stopping rule, i.e. another way of choosing the number of iterations, which is ignored for TFP."),
    ("pyro / numpyro / pymc / tfp", "guide / variational family choice (AutoNormal vs AutoMultivariateNormal, AutoIAFNormal, custom guides, surrogate family, PyMC method='advi' vs 'fullrank_advi')", "Per the analysis definition, guide choice doesn't matter."),
    ("pyro / numpyro", "guide structural arguments: rank, create_plates, prefix, hidden_dims, num_flows, batch_ndims, skip_deterministic, scale_constraint, hide / expose blocks", "These define which guide family is used, not its starting values; covered by 'guide choice doesn't matter'."),
    ("stan", "algorithm ('meanfield' vs 'fullrank')", "Variational family choice; treated the same way as guide choice, which doesn't matter."),
    ("pyro / numpyro", "JIT variants of the default ELBOs (JitTrace_ELBO, JitTraceMeanField_ELBO)", "JIT compilation computes the same objective as Trace_ELBO / TraceMeanField_ELBO."),
    ("pyro / numpyro", "ELBO vectorize_particles, max_plate_nesting, max_iarange_nesting, strict_enumeration_warning, ignore_jit_warnings, jit_options, retain_graph", "Computational/bookkeeping options; they don't change the objective being optimized."),
    ("pyro", "SVI num_samples, num_steps (deprecated constructor args)", "Control posterior sampling after the fit / a deprecated iteration count, not the fit itself."),
    ("pyro", "torch optimizer implementation flags (foreach, fused, capturable, differentiable)", "Implementation choices; same update rule."),
    ("numpyro", "SVI **static_kwargs", "Passed through to the model as data/arguments, not fit settings."),
    ("pyro / numpyro", "arguments to svi.step(...) / extra positional or keyword arguments to svi.run(...)", "Forwarded to the model and guide (data, mini-batches, annealing factors), not fit settings."),
    ("numpyro", "svi.run rng_key, progress_bar, stable_update, forward_mode_differentiation", "Random seed, display, and numerical-safety/AD-mode options that don't change the objective or its starting point. (stable_update only skips updates that would produce NaN/Inf.)"),
    ("tensorflow_probability", "trace_fn, jit_compile, seed, name, trainable_variables", "Logging, compilation, seed, and naming. trainable_variables is in practice the surrogate's own variables, which is what TFP uses automatically."),
    ("tensorflow_probability", "optimizer name, jit_compile", "Naming and compilation only."),
    ("tensorflow_probability", "hand-built surrogate posteriors (tf.Variable starting values)", "Starting values written directly into a custom surrogate are part of the guide definition and can't be read statically; treated like guide choice. Only initial_* arguments to TFP's surrogate builders are checked."),
    ("pymc", "random_seed, progressbar, model, score, fn_kwargs, compile_kwargs", "Seed, display, and compilation options."),
    ("pymc", "callbacks other than CheckParametersConvergence (e.g. pm.callbacks.Tracker, or a callback passed by variable name)", "Trackers only record values and don't stop the fit early. A callback held in a variable is assumed to be a tracker."),
    ("pymc", "more_replacements", "Mini-batch data replacement: part of the model/data specification, not an optimizer setting."),
    ("pymc", "tf_n_mc, test_optimizer, more_tf_params", "Only used by operator-variational methods (SVGD/ASVGD), not ADVI."),
    ("pymc", "random_seed and model arguments to pm.ADVI(...) / pm.FullRankADVI(...)", "Seed and model selection only."),
    ("stan", "output_samples / draws", "Number of draws taken from the fitted approximation afterwards; doesn't change the fit."),
    ("stan", "seed, data, output_dir, sig_figs, save_latent_dynamics, save_profile, require_converged, show_console, refresh, time_fmt, timeout, pars, sample_file, diagnostic_file, verbose", "Seed, data, output, logging, and error-handling options."),
    ("stan", "CmdStanModel / StanModel constructor arguments", "Model compilation options."),
]

# --------------------------------------------------------------------------
# Expression helpers
# --------------------------------------------------------------------------


class Unresolved(Exception):
    pass


def parse_expr(text: str | None) -> ast.expr | None:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def last_name(node: ast.expr) -> str:
    text = ast.unparse(node)
    return text.rsplit(".", 1)[-1]


def number(node: ast.expr | None) -> float:
    """Evaluate a numeric literal expression (1e-3, 10**-3, 1/1000, -2).
    Raises Unresolved for anything that isn't a pure literal."""
    if node is None:
        raise Unresolved("missing value")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = number(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = number(node.left), number(node.right)
        ops = {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right}
        if type(node.op) in ops:
            return ops[type(node.op)]
        if isinstance(node.op, ast.Div) and right:
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise Unresolved(ast.unparse(node))


def alternatives(node: ast.expr | None) -> list[ast.expr | None]:
    """`a if cond else b` (from the source or from tracing) -> [a, b]."""
    if isinstance(node, ast.IfExp):
        if isinstance(node.test, ast.Constant):
            return alternatives(node.body if node.test.value else node.orelse)
        return alternatives(node.body) + alternatives(node.orelse)
    return [node]


def any_alternative(node: ast.expr | None, predicate) -> bool:
    """True if any alternative satisfies predicate; Unresolved if none does and
    some can't be read; False if all can be read and none does."""
    unreadable = None
    for option in alternatives(node):
        try:
            if predicate(option):
                return True
        except Unresolved as exc:
            unreadable = exc
    if unreadable is not None:
        raise unreadable
    return False


def is_literal(node: ast.expr) -> bool:
    return isinstance(node, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.UnaryOp, ast.BinOp))


def same_number(node: ast.expr | None, expected: float) -> bool:
    return any_alternative(node, lambda n: math.isclose(number(n), expected, rel_tol=1e-9, abs_tol=1e-15))


def boolean(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return bool(node.value)
    raise Unresolved(ast.unparse(node))


def is_none(node: ast.expr | None) -> bool:
    return any(n is None or (isinstance(n, ast.Constant) and n.value is None) for n in alternatives(node))


def call_args(node: ast.Call) -> tuple[list[ast.expr], dict[str, ast.expr], bool]:
    """(positional args, keyword args, has **kwargs spread)."""
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
    spread = any(kw.arg is None for kw in node.keywords)
    return list(node.args), kwargs, spread


TRACED: dict[str, Any] = {}


def stored_args(row: dict[str, Any]) -> tuple[list[ast.expr | None], dict[str, ast.expr | None], bool]:
    """Positional and keyword arguments of a stored call row, as AST nodes.
    Uses values traced through the repository's source when available, else
    the pipeline's in-file alias resolutions (__resolved_*)."""
    traced = TRACED.get(str(row.get("id")))
    if traced is not None:
        kwargs = {k: parse_expr(v) for k, v in traced["keywords"].items() if k != "**"}
        return [parse_expr(p) for p in traced["positional"]], kwargs, "**" in traced["keywords"]
    kw_raw = json.loads(row.get("keyword_arguments") or "{}")
    pos_raw = json.loads(row.get("positional_arguments") or "[]")
    kwargs: dict[str, ast.expr | None] = {}
    for name, value in kw_raw.items():
        if name.startswith("__") or name == "**":
            continue
        resolved = kw_raw.get(f"__resolved_{name}")
        kwargs[name] = parse_expr(resolved or value)
    return [parse_expr(p) for p in pos_raw], kwargs, "**" in kw_raw


class Verdict:
    def __init__(self) -> None:
        self.non_default: list[str] = []
        self.unresolved: list[str] = []
        self.ignored: set[str] = set()

    def bad(self, reason: str) -> None:
        if reason not in self.non_default:
            self.non_default.append(reason)

    def unknown(self, reason: str) -> None:
        if reason not in self.unresolved:
            self.unresolved.append(reason)

    def check(self, label: str, test) -> None:
        """Run test(); False -> non-default, Unresolved -> unresolved."""
        try:
            if not test():
                self.bad(label)
        except Unresolved as exc:
            self.unknown(f"{label}: can't read {exc}")

    @property
    def classification(self) -> str:
        if self.non_default:
            return "non_default"
        if self.unresolved:
            return "unresolved"
        return "default"


def dict_items(node: ast.expr | None) -> dict[str, ast.expr]:
    """Literal dict or dict(...) call -> {key: value node}. Raises Unresolved otherwise."""
    if isinstance(node, ast.Dict):
        items = {}
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                raise Unresolved(ast.unparse(node))
            items[key.value] = value
        return items
    if isinstance(node, ast.Call) and last_name(node.func) == "dict" and not node.args:
        _pos, kwargs, spread = call_args(node)
        if spread:
            raise Unresolved(ast.unparse(node))
        return kwargs
    raise Unresolved(ast.unparse(node) if node is not None else "missing")


# --------------------------------------------------------------------------
# Optimizer and loss rules
# --------------------------------------------------------------------------

SCHEDULE_MARKERS = ("schedule", "decay", "onecycle", "one_cycle", "cosine", "warmup", "piecewise", "polynomial", "cyclic", "lr_lambda")

# --------------------------------------------------------------------------
# Rule sets: which values count as default.
#   strict:   only the package default (or the analysis's choice where the
#             package has none) -- see CHECKED_PARAMETERS.
#   vignette: additionally, any value the developers use or suggest in their own
#             documentation, tutorials, or example scripts -- see VIGNETTE_SOURCES.
# --------------------------------------------------------------------------

STRICT_RULES: dict[str, Any] = {
    "pyro_lr": {1e-3},
    "pyro_betas": {(0.9, 0.999)},
    "pyro_any_lr_decay": False,
    "pyro_schedulers": set(),
    "pyro_other_optimizers": {},
    "elbos": {"Trace_ELBO", "TraceMeanField_ELBO", "JitTrace_ELBO", "JitTraceMeanField_ELBO"},
    "pyro_num_particles": {1},
    "numpyro_num_particles": {1},
    "pyro_init_loc_extra": set(),
    "numpyro_init_loc_extra": set(),
    "pyro_init_scale": {0.1},
    "numpyro_init_scale": {0.1},
    "numpyro_step": {5e-4},
    "numpyro_schedules_ok": False,
    "numpyro_optax_adam_lr": set(),
    "tfp_lr": {0.001},
    "tfp_sample_size": {1},
    "pymc_n": {10000},
    "pymc_optimizers": {("adagrad_window", 0.001)},
    "pymc_convergence_ok": False,
}

VIGNETTE_RULES: dict[str, Any] = {
    **STRICT_RULES,
    "pyro_lr": {1e-3, 1e-4, 0.01, 0.1, 0.2, 0.05, 0.03, 0.02, 0.006, 0.005, 0.003, 0.0005, 0.0003},
    "pyro_betas": {(0.9, 0.999), (0.8, 0.99), (0.9, 0.99), (0.93, 0.999)},
    "pyro_any_lr_decay": True,
    "pyro_schedulers": {"ExponentialLR", "MultiStepLR"},
    "pyro_other_optimizers": {"AdagradRMSProp": [{"eta": 0.2, "t": 0.05}, {"eta": 4.5, "t": 0.1}]},
    "elbos": STRICT_RULES["elbos"] | {"TraceEnum_ELBO", "JitTraceEnum_ELBO", "TraceGraph_ELBO", "JitTraceGraph_ELBO", "TraceTMC_ELBO"},
    "pyro_num_particles": {1, 16, 20, 2000},
    "numpyro_num_particles": {1, 10, 100, 1000},
    "pyro_init_loc_extra": {"init_to_feasible", "init_to_median", "init_to_mean"},
    "numpyro_init_loc_extra": {"init_to_median"},
    "pyro_init_scale": {0.1, 0.01, 0.05},
    "numpyro_step": {5e-4, 0.01, 0.005, 0.003, 0.001, 0.0003},
    "numpyro_schedules_ok": True,
    "numpyro_optax_adam_lr": {0.2, 0.1, 0.01},
    "tfp_lr": {0.001, 0.1, 0.05, 0.01},
    "tfp_sample_size": {1, 2, 4, 10, 16},
    "pymc_n": {10000, 100, 300, 400, 20000, 30000, 50000, 100000},
    "pymc_optimizers": {("adagrad_window", 0.001), ("adagrad_window", 0.01), ("adamax", 0.1), ("sgd", 0.01)},
    "pymc_convergence_ok": True,
}

RULES: dict[str, Any] = STRICT_RULES

PYRO = "https://pyro.ai/examples/"
NUMPYRO_EX = "https://num.pyro.ai/en/stable/examples/"
NUMPYRO_TUT = "https://num.pyro.ai/en/stable/tutorials/"
TFP = "https://www.tensorflow.org/probability/examples/"
PYMC_VI = "https://www.pymc.io/projects/examples/en/latest/variational_inference/"

VIGNETTE_SOURCES: list[tuple[str, str, str, str]] = [
    # (package, parameter, value accepted as default, where the developers use or suggest it)
    ("pyro", "Adam/ClippedAdam lr", "1e-4", f"recommended in SVI Part IV ({PYRO}svi_part_iv.html); AIR tutorial"),
    ("pyro", "Adam/ClippedAdam lr", "0.01", f"built-in default of pyro.contrib.gp.util.train; Modules ({PYRO}modules.html), Bayesian Regression II, Boosting BBVI, JIT, Tracking 1D"),
    ("pyro", "Adam/ClippedAdam lr", "0.1", f"GMM ({PYRO}gmm.html), Stable, autoname examples"),
    ("pyro", "Adam/ClippedAdam lr", "0.2", f"GMM ({PYRO}gmm.html)"),
    ("pyro", "Adam/ClippedAdam lr", "0.05", f"Dirichlet Process Mixture ({PYRO}dirichlet_process_mixture.html), Bayesian Regression II, Stable"),
    ("pyro", "Adam/ClippedAdam lr", "0.03", f"Bayesian Regression I ({PYRO}bayesian_regression.html)"),
    ("pyro", "Adam/ClippedAdam lr", "0.02", f"Introduction to Pyro ({PYRO}intro_long.html), EasyGuide, EKF"),
    ("pyro", "Adam/ClippedAdam lr", "0.006", f"scANVI ({PYRO}scanvi.html)"),
    ("pyro", "Adam/ClippedAdam lr", "0.005", f"Predictive with Deterministic ({PYRO}predictive_deterministic.html), Working Memory"),
    ("pyro", "Adam/ClippedAdam lr", "0.003", f"Introduction Part II ({PYRO}intro_part_ii.html)"),
    ("pyro", "Adam/ClippedAdam lr", "0.0005", f"SVI Part I ({PYRO}svi_part_i.html), SVI Part III"),
    ("pyro", "Adam/ClippedAdam lr", "0.0003", f"SS-VAE ({PYRO}ss-vae.html)"),
    ("pyro", "Adam/ClippedAdam betas", "(0.8, 0.99)", f"GMM ({PYRO}gmm.html)"),
    ("pyro", "Adam/ClippedAdam betas", "(0.9, 0.99)", f"Stable ({PYRO}stable.html)"),
    ("pyro", "Adam/ClippedAdam betas", "(0.93, 0.999)", f"SVI Part III ({PYRO}svi_part_iii.html)"),
    ("pyro", "ClippedAdam lrd", "any value <= 1 (learning-rate decay)", f"'consider using a decaying learning rate', SVI Part IV ({PYRO}svi_part_iv.html); Workflow, Logistic Growth, Stable"),
    ("pyro", "learning-rate scheduler", "ExponentialLR or MultiStepLR wrapping Adam/ClippedAdam (inner lr checked as above)", f"SVI Part IV decay advice; Working Memory ({PYRO}working_memory.html); scANVI ({PYRO}scanvi.html)"),
    ("pyro", "optimizer", "AdagradRMSProp with eta=0.2, t=0.05 or eta=4.5, t=0.1", "examples/sparse_gamma_def.py (https://github.com/pyro-ppl/pyro/blob/dev/examples/sparse_gamma_def.py)"),
    ("pyro / numpyro", "loss", "TraceEnum_ELBO, TraceGraph_ELBO, TraceTMC_ELBO (incl. JIT variants)", f"GMM ({PYRO}gmm.html), SVI Part III, AIR, DMM/HMM examples; NumPyro GMM ({NUMPYRO_TUT}gmm.html), NNX example"),
    ("pyro", "num_particles", "16, 20, 2000", f"SVI with Flow Guides ({PYRO}svi_flow_guide.html); Reconciling Experts; capture-recapture example"),
    ("pyro", "guide init_loc_fn", "init_to_feasible, init_to_median, init_to_mean (any autoguide)", f"package defaults; Bayesian Regression II ({PYRO}bayesian_regression_ii.html); sparse_gamma_def example"),
    ("pyro", "guide init_scale", "0.01, 0.05", f"Logistic Growth ({PYRO}logistic-growth.html), Workflow; NeuTra example"),
    ("numpyro", "Adam step_size", "0.01, 0.005, 0.003, 0.001, 0.0003", f"Covtype ({NUMPYRO_EX}covtype.html), Thompson Sampling, NeuTra, Zero-Inflated Poisson, HMCECS, Variationally Inferred Parameterization"),
    ("numpyro", "Adam step_size", "a learning-rate schedule", f"HSGP n-D example ({NUMPYRO_TUT}hsgp_nd_example.html)"),
    ("numpyro", "optimizer", "optax.adam with learning_rate 0.2, 0.1, 0.01 or a schedule", f"GMM ({NUMPYRO_TUT}gmm.html); toy mixture enumeration example; CVAE-Flax example"),
    ("numpyro", "num_particles", "10, 100, 1000", "Variationally Inferred Parameterization; minipyro and DAIS examples"),
    ("numpyro", "guide init_loc_fn", "init_to_median", f"HSGP n-D example ({NUMPYRO_TUT}hsgp_nd_example.html)"),
    ("tensorflow_probability", "Adam learning_rate", "0.1", f"every fit_surrogate_posterior API example; STS CO2 & Electricity ({TFP}Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand), Modeling with JointDistribution"),
    ("tensorflow_probability", "Adam learning_rate", "0.05", f"Probabilistic PCA ({TFP}Probabilistic_PCA)"),
    ("tensorflow_probability", "Adam learning_rate", "0.01", f"VI and Joint Distributions ({TFP}Variational_Inference_and_Joint_Distributions), Linear Mixed Effects VI"),
    ("tensorflow_probability", "sample_size", "2, 4, 10, 16", f"Linear Mixed Effects VI; VI and Joint Distributions; Modeling with JointDistribution ({TFP}Modeling_with_JointDistribution)"),
    ("pymc", "n", "100, 300, 400, 20000", f"Variational API Quickstart ({PYMC_VI}variational_api_quickstart.html)"),
    ("pymc", "n", "30000", f"Bayesian Neural Network ADVI ({PYMC_VI}bayesian_neural_network_advi.html)"),
    ("pymc", "n", "50000", "Factor Analysis (https://www.pymc.io/projects/examples/en/latest/case_studies/factor_analysis.html)"),
    ("pymc", "n", "100000", f"GLM Mini-batch ADVI ({PYMC_VI}GLM-hierarchical-advi-minibatch.html)"),
    ("pymc", "obj_optimizer", "adagrad_window(learning_rate=0.01), adamax(learning_rate=0.1), sgd(learning_rate=0.01)", f"Variational API Quickstart ({PYMC_VI}variational_api_quickstart.html)"),
    ("pymc", "callbacks", "CheckParametersConvergence (any settings)", f"Variational API Quickstart; GLM Mini-batch ADVI ({PYMC_VI}GLM-hierarchical-advi-minibatch.html)"),
    ("stan", "all variational settings", "no alternatives found (CmdStanPy's VI examples use the defaults)", "https://github.com/stan-dev/cmdstanpy/tree/develop/docsrc/users-guide/examples"),
]


def in_values(node: ast.expr | None, values) -> bool:
    return any_alternative(node, lambda n: any(math.isclose(number(n), x, rel_tol=1e-9, abs_tol=1e-15) for x in values))


def is_schedule(node: ast.expr) -> bool:
    return isinstance(node, ast.Lambda) or (
        isinstance(node, ast.Call) and any(m in ast.unparse(node.func).lower() for m in SCHEDULE_MARKERS)
    )


def check_step_size(v: Verdict, label: str, node: ast.expr, accepted, schedules_ok: bool = False) -> None:
    """A literal is compared with the accepted values; a schedule (lambda or a
    call to a decay/schedule function) is non-default unless the rule set allows
    schedules; anything else, e.g. float(args.lr) or cfg.get('lr'), is a value
    lookup that can't be read statically."""
    text = ast.unparse(node)
    if is_schedule(node):
        if not schedules_ok:
            v.bad(f"{label} schedule {text}")
        return
    v.check(f"{label}={text}", lambda: in_values(node, accepted))


PYRO_ADAM_DEFAULTS = {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.0}
PYRO_CLIPPED_EXTRA = {"clip_norm": 10.0, "lrd": 1.0}
PYRO_IGNORED_OPT_KEYS = {"foreach", "fused", "capturable", "differentiable"}


def branching(check):
    """Let a check handle `a if cond else b` (written in the source, or several
    traced values): a constant condition picks its branch; otherwise every
    branch is checked, and the setting is default if any branch is default."""
    def wrapper(v: Verdict, node, *args):
        if not isinstance(node, ast.IfExp):
            return check(v, node, *args)
        if isinstance(node.test, ast.Constant):
            return wrapper(v, node.body if node.test.value else node.orelse, *args)
        outcomes = []
        for branch in (node.body, node.orelse):
            sub = Verdict()
            wrapper(sub, branch, *args)
            outcomes.append(sub)
            v.ignored |= sub.ignored
        # Several possible values: default if any of them is default;
        # non-default only if all are; otherwise unresolved.
        kinds = {o.classification for o in outcomes}
        if "default" in kinds:
            return
        for o in outcomes:
            for reason in (o.non_default if kinds == {"non_default"} else o.unresolved):
                (v.bad if kinds == {"non_default"} else v.unknown)(reason)
    return wrapper


@branching
def check_pyro_optimizer(v: Verdict, node: ast.expr | None) -> None:
    if node is None:
        v.unknown("optimizer: not found")
        return
    if not isinstance(node, ast.Call):
        v.unknown(f"optimizer: can't read {ast.unparse(node)}")
        return
    name = last_name(node.func)
    pos, kwargs, _spread = call_args(node)
    if name == "PyroOptim" and pos and last_name(pos[0]) in {"Adam", "ClippedAdam"}:
        name, pos = last_name(pos[0]), pos[1:]
    if name == "PyroLRScheduler" and pos and last_name(pos[0]) in RULES["pyro_schedulers"]:
        name, pos = last_name(pos[0]), pos[1:]
    if name in RULES["pyro_schedulers"]:
        check_pyro_scheduler(v, name, pos[0] if pos else kwargs.get("optim_args"))
        return
    if name in RULES["pyro_other_optimizers"]:
        check_pyro_other_optimizer(v, name, pos[0] if pos else kwargs.get("optim_args"))
        return
    if name not in {"Adam", "ClippedAdam"}:
        v.bad(f"optimizer {name}")
        return
    if not is_none(kwargs.get("clip_args")):
        v.bad("optimizer clip_args set")
    optim_args = pos[0] if pos else kwargs.get("optim_args")
    if isinstance(optim_args, ast.Lambda):
        v.bad("per-parameter optimizer settings (callable optim_args)")
        return
    try:
        items = dict_items(optim_args)
    except Unresolved as exc:
        v.unknown(f"optimizer args: can't read {exc}")
        return
    check_pyro_adam_settings(v, name, items)


def check_pyro_adam_settings(v: Verdict, name: str, items: dict[str, ast.expr]) -> None:
    defaults = dict(PYRO_ADAM_DEFAULTS)
    if name == "ClippedAdam":
        defaults.update(PYRO_CLIPPED_EXTRA)
    for key, value in items.items():
        if key in PYRO_IGNORED_OPT_KEYS:
            v.ignored.add(f"optimizer {key}")
        elif key == "lr":
            check_step_size(v, "optimizer lr", value, RULES["pyro_lr"])
        elif key == "lrd" and name == "ClippedAdam" and RULES["pyro_any_lr_decay"]:
            v.check(f"optimizer lrd={ast.unparse(value)}", lambda value=value: number(value) <= 1.0)
        elif key in defaults:
            v.check(f"optimizer {key}={ast.unparse(value)}", lambda value=value, key=key: same_number(value, defaults[key]))
        elif key == "betas":
            def betas_ok(value=value):
                if not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) != 2:
                    raise Unresolved(ast.unparse(value))
                pair = (number(value.elts[0]), number(value.elts[1]))
                return any(math.isclose(pair[0], a) and math.isclose(pair[1], b) for a, b in RULES["pyro_betas"])
            v.check(f"optimizer betas={ast.unparse(value)}", betas_ok)
        elif key in {"amsgrad", "maximize"}:
            v.check(f"optimizer {key}={ast.unparse(value)}", lambda value=value: not boolean(value))
        else:
            v.unknown(f"optimizer: unrecognized setting {key!r}")


def check_pyro_scheduler(v: Verdict, name: str, args: ast.expr | None) -> None:
    """ExponentialLR / MultiStepLR({'optimizer': Adam, 'optim_args': {...}, ...}):
    the decay itself is accepted; the wrapped optimizer is checked like any other."""
    try:
        items = dict_items(args)
    except Unresolved as exc:
        v.unknown(f"{name} args: can't read {exc}")
        return
    inner = items.get("optimizer")
    inner_name = last_name(inner) if inner is not None else ""
    if inner_name not in {"Adam", "ClippedAdam"}:
        v.bad(f"{name} wrapping {ast.unparse(inner) if inner is not None else 'unknown optimizer'}")
        return
    try:
        inner_items = dict_items(items.get("optim_args"))
    except Unresolved as exc:
        v.unknown(f"{name} optim_args: can't read {exc}")
        return
    v.ignored.add(f"optimizer {name} decay settings")
    check_pyro_adam_settings(v, inner_name, inner_items)


def check_pyro_other_optimizer(v: Verdict, name: str, args: ast.expr | None) -> None:
    """A non-Adam optimizer used in a developer example: default only with the
    exact settings the example uses."""
    try:
        items = dict_items(args)
        values = {key: number(value) for key, value in items.items()}
    except Unresolved as exc:
        v.unknown(f"{name} args: can't read {exc}")
        return
    for allowed in RULES["pyro_other_optimizers"][name]:
        if values.keys() == allowed.keys() and all(math.isclose(values[k], allowed[k]) for k in allowed):
            return
    v.bad(f"optimizer {name}{values}")


NUMPYRO_ADAM_DEFAULTS = (("b1", 0.9), ("b2", 0.999), ("eps", 1e-8))


@branching
def check_numpyro_optimizer(v: Verdict, node: ast.expr | None) -> None:
    if node is None:
        v.unknown("optimizer: not found")
        return
    if not isinstance(node, ast.Call):
        v.unknown(f"optimizer: can't read {ast.unparse(node)}")
        return
    func = ast.unparse(node.func)
    name = last_name(node.func)
    pos, kwargs, spread = call_args(node)
    if "optax" in func and name == "adam" and RULES["numpyro_optax_adam_lr"]:
        lr = pos[0] if pos else kwargs.get("learning_rate")
        if lr is None:
            v.unknown("optax.adam learning_rate: not found")
        else:
            check_step_size(v, "optax.adam learning_rate", lr, RULES["numpyro_optax_adam_lr"], RULES["numpyro_schedules_ok"])
        return
    if "optim_to_numpyro" in func or "optax_to_numpyro" in func:
        if pos and isinstance(pos[0], ast.Call):
            check_numpyro_optimizer(v, pos[0])
            return
    if name != "Adam" or "optax" in func:
        v.bad(f"optimizer {func}")
        return
    step = pos[0] if pos else kwargs.get("step_size")
    if step is None:
        v.unknown("optimizer step_size: " + ("passed via **kwargs" if spread else "not found"))
    else:
        check_step_size(v, "optimizer step_size", step, RULES["numpyro_step"], RULES["numpyro_schedules_ok"])
    for index, (key, default) in enumerate(NUMPYRO_ADAM_DEFAULTS, start=1):
        value = pos[index] if len(pos) > index else kwargs.get(key)
        if value is not None:
            v.check(f"optimizer {key}={ast.unparse(value)}", lambda value=value, default=default: same_number(value, default))


@branching
def check_pyro_or_numpyro_optimizer(v: Verdict, node: ast.expr | None) -> None:
    """Package couldn't be told apart from imports; the optimizer's call style
    decides: a dict of settings is Pyro, a step size is NumPyro."""
    if isinstance(node, ast.Call):
        pos, kwargs, _ = call_args(node)
        first = pos[0] if pos else None
        if "optax" in ast.unparse(node.func):
            check_numpyro_optimizer(v, node)
            return
        if "optim_args" in kwargs or isinstance(first, ast.Dict) or (
            isinstance(first, ast.Call) and last_name(first.func) == "dict"
        ) or last_name(node.func) in {"ClippedAdam", "PyroOptim"}:
            check_pyro_optimizer(v, node)
            return
        if "step_size" in kwargs or (first is not None and is_literal(first)):
            check_numpyro_optimizer(v, node)
            return
        if last_name(node.func) != "Adam":
            v.bad(f"optimizer {ast.unparse(node.func)}")
            return
    v.unknown(f"optimizer: can't tell Pyro vs NumPyro style from {ast.unparse(node) if node is not None else 'missing'}")


IGNORED_ELBO_KWARGS = {
    "vectorize_particles", "max_plate_nesting", "max_iarange_nesting", "strict_enumeration_warning",
    "ignore_jit_warnings", "jit_options", "retain_graph",
}


@branching
def check_elbo(v: Verdict, node: ast.expr | None, package: str) -> None:
    if node is None:
        v.unknown("loss: not found")
        return
    if isinstance(node, ast.Constant) and node.value == "ELBO":
        v.ignored.add("loss 'ELBO' (pre-0.2 Pyro name for Trace_ELBO)")
        return
    if not isinstance(node, ast.Call):
        v.unknown(f"loss: can't read {ast.unparse(node)}")
        return
    name = last_name(node.func)
    if name not in RULES["elbos"]:
        v.bad(f"loss {name}")
        return
    if name.startswith("Jit"):
        v.ignored.add("loss JIT variant")
    pos, kwargs, spread = call_args(node)
    particles = pos[0] if pos else kwargs.get("num_particles")
    if particles is not None:
        allowed = (RULES["numpyro_num_particles"] if package == "numpyro"
                   else RULES["pyro_num_particles"] if package == "pyro"
                   else RULES["pyro_num_particles"] | RULES["numpyro_num_particles"])
        v.check(f"loss num_particles={ast.unparse(particles)}", lambda: in_values(particles, allowed))
    for key, value in kwargs.items():
        if key == "num_particles":
            continue
        if key in IGNORED_ELBO_KWARGS:
            v.ignored.add(f"loss {key}")
        elif key == "multi_sample_guide":
            v.check(f"loss multi_sample_guide={ast.unparse(value)}", lambda value=value: not boolean(value))
        else:
            v.unknown(f"loss: unrecognized setting {key!r}")
    if spread:
        v.unknown("loss: settings passed via **kwargs")


# --------------------------------------------------------------------------
# Initial-value rules for guides / constructors / run calls
# --------------------------------------------------------------------------

PYRO_INIT_LOC_DEFAULT = {"AutoNormal": "init_to_feasible"}  # every other Pyro autoguide: init_to_median
GUIDE_STRUCTURAL_KWARGS = {
    "rank", "create_plates", "prefix", "hidden_dim", "hidden_dims", "num_flows", "batch_ndims",
    "skip_deterministic", "scale_constraint", "hide", "expose", "hide_fn", "expose_fn", "num_particles",
}


def check_guide_constructor(v: Verdict, node: ast.Call, package: str) -> None:
    name = last_name(node.func)
    pos, kwargs, spread = call_args(node)
    if not name.startswith("Auto"):
        v.ignored.add("custom guide (guide choice)")
        return
    if len(pos) > 1:
        v.unknown(f"guide {name}: positional arguments beyond the model")
    if spread:
        v.unknown(f"guide {name}: settings passed via **kwargs")
    is_numpyro = package == "numpyro"
    for key, value in kwargs.items():
        if key in {"init_loc_fn", "init_strategy"} and len(alternatives(value)) > 1:
            outcomes = []
            for option in alternatives(value):
                sub = Verdict()
                check_guide_constructor(sub, ast.Call(func=node.func, args=[], keywords=[ast.keyword(arg=key, value=option)]), package)
                outcomes.append(sub.classification)
            if "default" not in outcomes:
                (v.bad if set(outcomes) == {"non_default"} else v.unknown)(f"guide {key}={ast.unparse(value)[:100]}")
        elif key in {"init_loc_fn", "init_strategy"}:
            if is_numpyro:
                expected = {"init_to_uniform"} | RULES["numpyro_init_loc_extra"]
            elif package == "pyro":
                expected = {PYRO_INIT_LOC_DEFAULT.get(name, "init_to_median")} | RULES["pyro_init_loc_extra"]
            else:  # pyro_or_numpyro: accept either package's own default
                expected = ({"init_to_uniform", PYRO_INIT_LOC_DEFAULT.get(name, "init_to_median")}
                            | RULES["pyro_init_loc_extra"] | RULES["numpyro_init_loc_extra"])
            text = ast.unparse(value)
            func = value.func if isinstance(value, ast.Call) else value
            # A named strategy taken from the rule set may carry settings in the
            # vignette rules (e.g. init_to_median(num_samples=25)); the package's
            # own default must be used as-is in the strict rules.
            settings_ok = RULES is VIGNETTE_RULES and last_name(func) != "init_to_value"
            is_default = last_name(func) in expected and (
                not isinstance(value, ast.Call) or not (value.args or value.keywords) or settings_ok)
            if is_default:
                continue
            if last_name(func).startswith("init_to_") or isinstance(value, ast.Lambda):
                v.bad(f"guide {key}={text}")
            else:
                v.unknown(f"guide {key}: can't read {text}")
        elif key == "init_scale":
            allowed = RULES["numpyro_init_scale"] if is_numpyro else RULES["pyro_init_scale"]
            v.check(f"guide init_scale={ast.unparse(value)}", lambda value=value, allowed=allowed: in_values(value, allowed))
        elif key.startswith("init"):
            v.bad(f"guide {key}={ast.unparse(value)}")
        elif key in GUIDE_STRUCTURAL_KWARGS:
            v.ignored.add(f"guide {key}")
        else:
            v.ignored.add(f"guide {key}")


NUMPYRO_RUN_IGNORED = {"progress_bar", "stable_update", "forward_mode_differentiation", "rng_key", "num_steps"}


def check_numpyro_run(v: Verdict, row: dict[str, Any]) -> None:
    _pos, kwargs, _spread = stored_args(row)
    for key, value in kwargs.items():
        if key in {"init_params", "init_state"}:
            if not is_none(value):
                v.bad(f"svi.run {key} set")
        elif key in NUMPYRO_RUN_IGNORED:
            v.ignored.add(f"svi.run {key}")
        else:
            v.ignored.add("svi.run model arguments")


# --------------------------------------------------------------------------
# Per-package configuration checks
# --------------------------------------------------------------------------


def guide_call_from_snippet(snippet: str) -> ast.Call | None:
    """The first Auto*(...) call in a snippet, wherever it sits
    (`guide = AutoNormal(m)`, `return AutoNormal(m)`, `guide=AutoNormal(m),`,
    `guide.add(AutoNormal(m))`). None if the snippet cuts the call off."""
    for match in re.finditer(r"\b(?:[\w.]+\.)?Auto\w+\s*\(", snippet):
        depth, start = 0, match.start()
        for index in range(match.end() - 1, len(snippet)):
            depth += {"(": 1, ")": -1}.get(snippet[index], 0)
            if depth == 0:
                node = parse_expr(snippet[start:index + 1])
                if isinstance(node, ast.Call):
                    return node
                break
    return None


def check_svi_group(v: Verdict, package: str, rel: dict[str, Any], settings: dict[str, Any] | None, members: list[dict[str, Any]]) -> None:
    optimizer = parse_expr(rel.get("optimizer_expression"))
    loss = parse_expr(rel.get("loss_expression"))
    if package == "pyro":
        check_pyro_optimizer(v, optimizer)
    elif package == "numpyro":
        check_numpyro_optimizer(v, optimizer)
    else:
        check_pyro_or_numpyro_optimizer(v, optimizer)
    check_elbo(v, loss, package)

    guide_nodes: list[tuple[ast.Call, str]] = []
    if settings is not None:
        pos, kwargs, _spread = stored_args(settings)
        guide = kwargs.get("guide") or (pos[1] if len(pos) > 1 else None)
        if isinstance(guide, ast.Call):
            guide_nodes.append((guide, settings["package"]))
        if not is_none(kwargs.get("loss_and_grads")):
            v.bad("SVI loss_and_grads set")
        for key in ("num_samples", "num_steps"):
            if key in kwargs:
                v.ignored.add(f"SVI {key}")
    for member in members:
        kind = member["call_kind"]
        if kind == "autoguide_constructor":
            traced = TRACED.get(str(member.get("id")))
            node = parse_expr(traced["call"]) if traced else guide_call_from_snippet(member["source_snippet"])
            if isinstance(node, ast.Call):
                guide_nodes.append((node, member["package"]))
            else:
                v.unknown(f"guide constructor not parseable: {member['source_snippet'][:80]}")
        elif kind == "svi_run_method":
            check_numpyro_run(v, member)
        elif kind == "svi_step_method":
            v.ignored.add("svi.step model arguments")
    for node, guide_package in guide_nodes:
        check_guide_constructor(v, node, guide_package)


TFP_KERAS_ADAM_DEFAULTS = {"beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-7, "beta1": 0.9, "beta2": 0.999}
TFP_ADAM_NON_DEFAULT_IF_SET = {"clipnorm", "clipvalue", "global_clipnorm", "weight_decay", "use_ema"}


@branching
def check_tfp_optimizer(v: Verdict, node: ast.expr | None) -> None:
    if node is None:
        v.unknown("optimizer: not found")
        return
    if not isinstance(node, ast.Call):
        v.unknown(f"optimizer: can't read {ast.unparse(node)}")
        return
    func = ast.unparse(node.func)
    name = last_name(node.func)
    if name not in {"Adam", "AdamOptimizer"} or "optax" in func:
        v.bad(f"optimizer {func}")
        return
    pos, kwargs, spread = call_args(node)
    lr = pos[0] if pos else kwargs.get("learning_rate")
    if lr is not None:
        check_step_size(v, "optimizer learning_rate", lr, RULES["tfp_lr"])
    if len(pos) > 1:
        v.unknown("optimizer: positional settings beyond learning_rate")
    if spread:
        v.unknown("optimizer: settings passed via **kwargs")
    for key, value in kwargs.items():
        if key == "learning_rate":
            continue
        if key in TFP_KERAS_ADAM_DEFAULTS:
            v.check(f"optimizer {key}={ast.unparse(value)}", lambda value=value, key=key: same_number(value, TFP_KERAS_ADAM_DEFAULTS[key]))
        elif key == "amsgrad":
            v.check(f"optimizer amsgrad={ast.unparse(value)}", lambda value=value: not boolean(value))
        elif key in TFP_ADAM_NON_DEFAULT_IF_SET:
            if not is_none(value) and not (isinstance(value, ast.Constant) and value.value is False):
                v.bad(f"optimizer {key}={ast.unparse(value)}")
        elif key in {"name", "jit_compile", "use_locking"}:
            v.ignored.add(f"optimizer {key}")
        else:
            v.unknown(f"optimizer: unrecognized setting {key!r}")


TFP_FIT_IGNORED = {"trace_fn", "jit_compile", "seed", "name", "trainable_variables", "convergence_criterion", "num_steps",
                   "target_log_prob_fn", "surrogate_posterior", "optimizer"}


def check_tfp_group(v: Verdict, rel: dict[str, Any], settings: dict[str, Any] | None, members: list[dict[str, Any]]) -> None:
    check_tfp_optimizer(v, parse_expr(rel.get("optimizer_expression")))
    if settings is not None:
        pos, kwargs, spread = stored_args(settings)
        sample_size = kwargs.get("sample_size") or (pos[6] if len(pos) > 6 else None)
        if sample_size is not None:
            v.check(f"sample_size={ast.unparse(sample_size)}", lambda: in_values(sample_size, RULES["tfp_sample_size"]))
        if kwargs.get("importance_sample_size") is not None:
            iss = kwargs["importance_sample_size"]
            v.check(f"importance_sample_size={ast.unparse(iss)}", lambda: same_number(iss, 1))
        for key in ("variational_loss_fn", "discrepancy_fn"):
            value = kwargs.get(key)
            if value is not None and last_name(value) != "kl_reverse":
                v.bad(f"{key}={ast.unparse(value)}")
        if spread:
            v.unknown("fit_surrogate_posterior: settings passed via **kwargs")
        for key in kwargs:
            if key in TFP_FIT_IGNORED:
                v.ignored.add(f"fit_surrogate_posterior {key}")
    for member in members:
        if member["call_kind"] == "surrogate_posterior_constructor":
            _pos, kwargs, spread = stored_args(member)
            for key, value in kwargs.items():
                if key.startswith("initial") and not is_none(value):
                    v.bad(f"surrogate {key} set")
            if spread:
                v.unknown("surrogate builder: settings passed via **kwargs")


PYMC_IGNORED = {"random_seed", "progressbar", "model", "score", "fn_kwargs", "compile_kwargs", "more_replacements",
                "tf_n_mc", "test_optimizer", "more_tf_params"}
PYMC_CHECKED = {"n", "method", "obj_n_mc", "obj_optimizer", "start", "start_sigma", "inf_kwargs", "callbacks",
                "total_grad_norm_constraint", "more_obj_params", "more_updates"}
PYMC_ADVI_METHODS = {"advi", "fullrank_advi", "ADVI", "FullRankADVI"}


@branching
def check_adagrad_window(v: Verdict, node: ast.expr) -> None:
    if not isinstance(node, ast.Call):
        if last_name(node) == "adagrad_window":  # passed uncalled -> its defaults
            return
        v.unknown(f"obj_optimizer: can't read {ast.unparse(node)}")
        return
    name = last_name(node.func)
    accepted_lrs = {lr for opt, lr in RULES["pymc_optimizers"] if opt == name}
    if not accepted_lrs:
        v.bad(f"obj_optimizer {ast.unparse(node.func)}")
        return
    pos, kwargs, spread = call_args(node)
    if pos:
        v.unknown("obj_optimizer: positional arguments")
    if spread:
        v.unknown("obj_optimizer: settings passed via **kwargs")
    # Only the learning rate is varied in the accepted configurations; every
    # other setting must stay at adagrad_window's own default (or be absent).
    lr = kwargs.get("learning_rate")
    if lr is None and name != "adagrad_window":
        v.bad(f"obj_optimizer {name} with its own default learning rate")
    elif lr is not None:
        v.check(f"{name} learning_rate={ast.unparse(lr)}", lambda: in_values(lr, accepted_lrs))
    for key, default in (("epsilon", 0.1), ("n_win", 10)):
        if key in kwargs:
            v.check(f"{name} {key}={ast.unparse(kwargs[key])}", lambda key=key, default=default: name == "adagrad_window" and same_number(kwargs[key], default))
    for key in kwargs:
        if key not in {"learning_rate", "epsilon", "n_win", "loss_or_grads", "params"}:
            v.unknown(f"obj_optimizer: unrecognized setting {key!r}")


def check_pymc_start_kwargs(v: Verdict, kwargs: dict[str, ast.expr | None], where: str) -> None:
    for key in ("start", "start_sigma"):
        if key in kwargs and not is_none(kwargs[key]):
            v.bad(f"{where} {key} set")


def check_pymc_group(v: Verdict, rel: dict[str, Any], settings: dict[str, Any] | None, members: list[dict[str, Any]]) -> None:
    if settings is None:
        v.unknown("fit call not found")
        return
    pos, kwargs, spread = stored_args(settings)
    is_pm_fit = settings["call_kind"] == "fit"
    n = kwargs.get("n") or (pos[0] if pos else None)
    if n is not None:
        v.check(f"n={ast.unparse(n)}", lambda: in_values(n, RULES["pymc_n"]))
    if is_pm_fit:
        method = kwargs.get("method") or (pos[1] if len(pos) > 1 else None)
        if method is not None:
            if isinstance(method, ast.Constant) and isinstance(method.value, str):
                if method.value not in PYMC_ADVI_METHODS:
                    v.bad(f"method={method.value!r} (not ADVI)")
            elif isinstance(method, ast.Call) and last_name(method.func) in PYMC_ADVI_METHODS:
                _mpos, mkwargs, _ = call_args(method)
                check_pymc_start_kwargs(v, mkwargs, "method")
            else:
                v.unknown(f"method: can't read {ast.unparse(method)}")
    obj_n_mc = parse_expr(rel.get("sample_or_mc_expression")) if rel.get("sample_or_mc_expression") else kwargs.get("obj_n_mc")
    if obj_n_mc is not None:
        v.check(f"obj_n_mc={ast.unparse(obj_n_mc)}", lambda: same_number(obj_n_mc, 1))
    optimizer = parse_expr(rel.get("optimizer_expression")) or kwargs.get("obj_optimizer")
    if optimizer is not None:
        check_adagrad_window(v, optimizer)
    check_pymc_start_kwargs(v, kwargs, "fit")
    inf_kwargs = kwargs.get("inf_kwargs")
    if inf_kwargs is not None:
        try:
            check_pymc_start_kwargs(v, dict_items(inf_kwargs), "inf_kwargs")
        except Unresolved as exc:
            v.unknown(f"inf_kwargs: can't read {exc}")
    callbacks = kwargs.get("callbacks")
    if callbacks is not None:
        if "Convergence" in ast.unparse(callbacks):
            if not RULES["pymc_convergence_ok"]:
                v.bad("CheckParametersConvergence callback (stops before n iterations)")
        else:
            v.ignored.add("callbacks (trackers)")
    for key in ("total_grad_norm_constraint", "more_obj_params", "more_updates"):
        if key in kwargs and not is_none(kwargs[key]):
            v.bad(f"{key} set")
    for key in kwargs:
        if key in PYMC_IGNORED:
            v.ignored.add(f"fit {key}")
        elif key not in PYMC_CHECKED:
            v.unknown(f"fit: unrecognized setting {key!r}")
    if spread:
        v.unknown("fit: settings passed via **kwargs")
    for member in members:
        if member["call_kind"] == "advi_constructor":
            _mpos, mkwargs, mspread = stored_args(member)
            check_pymc_start_kwargs(v, mkwargs, "ADVI()")
            if mspread:
                v.unknown("ADVI(): settings passed via **kwargs")


STAN_DEFAULTS = {
    "iter": 10000, "grad_samples": 1, "elbo_samples": 100, "eta": 1.0, "adapt_iter": 50,
    "tol_rel_obj": 0.01, "eval_elbo": 100,
}
STAN_IGNORED = {
    "algorithm", "output_samples", "draws", "seed", "data", "output_dir", "sig_figs", "save_latent_dynamics",
    "save_profile", "require_converged", "show_console", "refresh", "time_fmt", "timeout", "pars", "sample_file",
    "diagnostic_file", "verbose",
}


def check_stan_group(v: Verdict, settings: dict[str, Any] | None) -> None:
    if settings is None:
        v.unknown("variational call not found")
        return
    pos, kwargs, spread = stored_args(settings)
    if len(pos) > 1:
        v.unknown("variational: positional arguments beyond data")
    for key, value in kwargs.items():
        if key in STAN_DEFAULTS:
            if key == "eta" and is_none(value):
                continue
            v.check(f"{key}={ast.unparse(value)}", lambda value=value, key=key: same_number(value, STAN_DEFAULTS[key]))
        elif key == "adapt_engaged":
            v.check(f"adapt_engaged={ast.unparse(value)}", lambda value=value: boolean(value))
        elif key in {"inits", "init"}:
            if not (is_none(value) or (isinstance(value, ast.Constant) and value.value == "random")):
                v.bad(f"{key}={ast.unparse(value)}")
        elif key in STAN_IGNORED:
            v.ignored.add(f"variational {key}")
        else:
            v.unknown(f"variational: unrecognized setting {key!r}")
    if spread:
        v.unknown("variational: settings passed via **kwargs")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def load(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    relevant = query_rows(conn, GITHUB_ADVI_RELEVANT_PARAMETER_ANALYSIS_QUERY)
    OVERRIDDEN.update(
        (r, p, int(l)) for r, p, l in conn.execute("SELECT repo_full_name, path, line_number FROM github_advi_parameter_overrides")
    )
    calls = {row["id"]: dict(row) for row in conn.execute("SELECT * FROM github_advi_calls")}
    true_positive = {
        row[0] for row in conn.execute(
            "SELECT call_id FROM github_advi_call_classifications WHERE usage_status='likely_true_positive'"
        )
    }
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for call_id, settings_id in conn.execute(
        "SELECT call_id, settings_call_id FROM github_advi_settings_companions WHERE settings_call_id IS NOT NULL"
    ):
        if call_id in true_positive and call_id in calls:
            groups[settings_id].append(calls[call_id])
    return relevant, calls, groups


OVERRIDDEN: set[tuple[str, str, int]] = set()


def traced_relevant(rel: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """The optimizer / loss / MC expressions re-read from the traced settings call."""
    pos, kwargs, _ = stored_args(settings)
    def pick(*names, index=None):
        for name in names:
            if kwargs.get(name) is not None:
                return ast.unparse(kwargs[name])
        if index is not None and len(pos) > index and pos[index] is not None:
            return ast.unparse(pos[index])
        return None
    rel = dict(rel)
    package = rel["package"]
    if package in {"pyro", "numpyro", "pyro_or_numpyro"}:
        rel["optimizer_expression"] = pick("optim", "optimizer", index=2) or rel.get("optimizer_expression")
        rel["loss_expression"] = pick("loss", index=3) or rel.get("loss_expression")
    elif package == "tensorflow_probability":
        rel["optimizer_expression"] = pick("optimizer", index=2) or rel.get("optimizer_expression")
    elif package == "pymc":
        rel["optimizer_expression"] = pick("obj_optimizer") or rel.get("optimizer_expression")
        rel["sample_or_mc_expression"] = pick("obj_n_mc") or rel.get("sample_or_mc_expression")
    return rel


def classify(rel: dict[str, Any], calls: dict[int, dict[str, Any]], groups: dict[int, list[dict[str, Any]]]) -> Verdict:
    v = Verdict()
    settings_id = int(rel["settings_source_call_id"]) if rel.get("settings_source_call_id") else None
    settings = calls.get(settings_id) if settings_id else None
    members = groups.get(settings_id, []) if settings_id else []
    package = rel["package"]
    if settings is not None and str(settings_id) in TRACED and (settings["repo_full_name"], settings["path"], settings["line_number"]) not in OVERRIDDEN:
        rel = traced_relevant(rel, settings)
    if package in {"pyro", "numpyro", "pyro_or_numpyro"}:
        check_svi_group(v, package, rel, settings, members)
    elif package == "tensorflow_probability":
        check_tfp_group(v, rel, settings, members)
    elif package == "pymc":
        check_pymc_group(v, rel, settings, members)
    elif package == "stan":
        check_stan_group(v, settings)
    else:
        v.unknown(f"no rules for package {package}")
    return v


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}" if whole else ""


def summarize(labels: list[str]) -> dict[str, object]:
    counts = Counter(labels)
    d, n, u = counts["default"], counts["non_default"], counts["unresolved"]
    total = d + n + u
    return {
        "default": d, "non_default": n, "unresolved": u, "total": total,
        "default_pct_of_resolved": pct(d, d + n),
        "default_pct_low": pct(d, total),
        "default_pct_high": pct(d + u, total),
    }


def repo_label(labels: list[str]) -> str:
    if "non_default" in labels:
        return "non_default"
    if "unresolved" in labels:
        return "unresolved"
    return "default"


def load_package_labels(path: Path) -> dict[tuple[str, str, int], str]:
    """package_labels.csv (from verify_package_labels) -> {(repo, path, line): package}.
    Calls whose package couldn't be confirmed map to "pyro_or_numpyro" (excluded)."""
    labels = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            package = row["verified_package"]
            labels[(row["repo_full_name"], row["path"], int(row["line_number"]))] = (
                package if package in {"pyro", "numpyro"} else "pyro_or_numpyro")
    return labels


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, overview: list[dict[str, object]], rules: str = "strict") -> None:
    def table(header: list[str], body: list[tuple]) -> list[str]:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in body]
        return lines

    if rules == "vignette":
        intro = [
            "# Vignette-informed default analysis",
            "",
            "Generated by `python -m github_advi_usage.strict_default_analysis --rules vignette`; don't edit by hand.",
            "",
            "A call site is **default** if every input that can change the fitted approximation is either at the",
            "package's own default (the strict rules below) or at a value the developers use or suggest in their own",
            "documentation, tutorials, or example scripts (the accepted values below). Anything else is non-default.",
            "Iteration count is ignored for Pyro, NumPyro and TFP. **unresolved**",
        ]
    else:
        intro = [
            "# Strict default analysis",
            "",
            "Generated by `python -m github_advi_usage.strict_default_analysis`; don't edit by hand.",
            "",
            "A call site is **default** only if every input that can change the fitted approximation is at the",
            "value the package uses automatically (iteration count ignored for Pyro, NumPyro and TFP). **unresolved**",
        ]
    intro += ["", "Covers NumPyro, PyMC, TensorFlow Probability and Stan. Pyro is excluded, as are call sites whose",
              "imports don't show whether they use Pyro or NumPyro.", ""]
    lines = [
        *intro,
        "means nothing checked is non-default but at least one value can't be read statically (e.g. a learning",
        "rate passed in as a variable); the range columns treat all unresolved sites as non-default (low) or",
        "default (high).",
        "",
        "## Files",
        "",
        "- `strict_overview.csv`: counts and default % by package group, per call site and per repository.",
        "- `strict_call_site_classification.csv`: one row per true-positive call site with its strict",
        "  classification, the main analysis's classification, and the reasons.",
        "- `main_vs_strict_crosstab.csv`: main-analysis vs strict classification counts.",
        "- `reason_counts.csv`: how often each non-default / unresolved reason occurs.",
        "- `checked_parameters.csv`, `ignored_parameters.csv`"
        + (", `vignette_accepted_values.csv`" if rules == "vignette" else "") + ": the rule tables below.",
        "",
        "## Results",
        "",
        *table(
            ["unit", "package group", "default", "non-default", "unresolved", "default % of resolved", "default % range"],
            [(r["unit"], r["package_group"], r["default"], r["non_default"], r["unresolved"],
              r["default_pct_of_resolved"], f"{r['default_pct_low']}-{r['default_pct_high']}") for r in overview],
        ),
        "",
        *(["## Values accepted as default because the developers use or suggest them", "",
           *table(["package", "parameter", "accepted value", "developer source"], without_pyro(VIGNETTE_SOURCES)), ""]
          if rules == "vignette" else []),
        "## Parameters checked" + (" (package defaults; always accepted)" if rules == "vignette" else ""),
        "",
        *table(["package", "parameter", "default value", "rule"], without_pyro(CHECKED_PARAMETERS)),
        "",
        "## Parameters ignored, and why",
        "",
        *table(["package", "parameter", "why ignored"], without_pyro(IGNORED_PARAMETERS)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument(
        "--rules", choices=("strict", "vignette"), default="strict",
        help="strict: only package defaults count as default. vignette: values the developers use or "
        "suggest in their own docs/tutorials/examples also count (see VIGNETTE_SOURCES).",
    )
    parser.add_argument(
        "--package-labels", type=Path, default=None,
        help="package_labels.csv from verify_package_labels: Pyro/NumPyro call sites take the package their "
        "file's imports confirm; unconfirmed ones are excluded.",
    )
    parser.add_argument(
        "--traced", type=Path, default=None,
        help="traced_values.json from trace_unresolved_values; values traced through each repo's source "
        "replace ones the scan couldn't read. Output goes to <rules>_default_reports_traced.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <db folder>/<db stem>/<rules>_default_reports[_traced].")
    args = parser.parse_args()
    global RULES
    RULES = VIGNETTE_RULES if args.rules == "vignette" else STRICT_RULES
    if args.traced is not None:
        TRACED.update(json.loads(args.traced.read_text()))
    suffix = "_traced" if args.traced is not None else ""
    out = args.out or args.db.parent / args.db.stem / f"{args.rules}_default_reports{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    relevant, calls, groups = load(conn)

    verified = load_package_labels(args.package_labels) if args.package_labels else None
    rows: list[dict[str, object]] = []
    for rel in relevant:
        if verified is not None:
            key = (rel["repo_full_name"], rel["path"], int(rel["line_number"]))
            if key in verified:
                # Pyro/NumPyro call sites take the package their file's imports confirm.
                rel = {**rel, "package": verified[key]}
        if rel["package"] in EXCLUDED_PACKAGES:
            continue
        v = classify(rel, calls, groups)
        current = final_from_relevant(rel)
        current = "default" if current == "default" else "needs_review" if current == "needs_review" else "adjusted"
        rows.append({
            "call_site_id": call_site_id(rel["repo_full_name"], rel["html_url"], rel["path"], rel["line_number"]),
            "repo_full_name": rel["repo_full_name"],
            "package": rel["package"],
            "package_group": PACKAGE_GROUPS.get(rel["package"], rel["package"]),
            "call_kind": rel["call_kind"],
            "github_permalink": rel["html_url"],
            "strict_classification": v.classification,
            "main_analysis_classification": current,
            "non_default_reasons": "; ".join(v.non_default),
            "unresolved_reasons": "; ".join(v.unresolved),
            "ignored_parameters_present": "; ".join(sorted(v.ignored)),
            "optimizer_expression": rel.get("optimizer_expression") or "",
            "loss_expression": rel.get("loss_expression") or "",
            "source_snippet": rel.get("source_snippet") or "",
        })

    write_csv(out / "strict_call_site_classification.csv", rows, list(rows[0].keys()))

    overview: list[dict[str, object]] = []
    for unit in ("call_site", "repository"):
        for group in [*GROUP_ORDER, "All packages"]:
            subset = [r for r in rows if group == "All packages" or r["package_group"] == group]
            if unit == "call_site":
                labels = [str(r["strict_classification"]) for r in subset]
            else:
                by_repo: dict[str, list[str]] = defaultdict(list)
                for r in subset:
                    by_repo[str(r["repo_full_name"])].append(str(r["strict_classification"]))
                labels = [repo_label(v) for v in by_repo.values()]
            overview.append({"unit": unit, "package_group": group, **summarize(labels)})
    write_csv(out / "strict_overview.csv", overview, list(overview[0].keys()))

    crosstab = Counter((r["main_analysis_classification"], r["strict_classification"]) for r in rows)
    write_csv(
        out / "main_vs_strict_crosstab.csv",
        [{"main_analysis": a, "strict": b, "call_sites": n} for (a, b), n in sorted(crosstab.items())],
        ["main_analysis", "strict", "call_sites"],
    )
    # Each reason is counted under its own type (a call site that is non-default
    # because of its learning rate can also have an unreadable guide setting).
    reasons = Counter()
    for r in rows:
        for field, reason_type in (("non_default_reasons", "non_default"), ("unresolved_reasons", "unresolved")):
            for generic in {reason.split("=", 1)[0].split(": can't read", 1)[0]
                            for reason in filter(None, str(r[field]).split("; "))}:
                reasons[(r["package_group"], reason_type, generic)] += 1
    write_csv(
        out / "reason_counts.csv",
        [{"package_group": g, "reason_type": t, "reason": reason, "call_sites": n}
         for (g, t, reason), n in reasons.most_common()],
        ["package_group", "reason_type", "reason", "call_sites"],
    )
    write_csv(
        out / "checked_parameters.csv",
        [dict(zip(("package", "parameter", "default_value", "rule"), row)) for row in without_pyro(CHECKED_PARAMETERS)],
        ["package", "parameter", "default_value", "rule"],
    )
    write_csv(
        out / "ignored_parameters.csv",
        [dict(zip(("package", "parameter", "why_ignored"), row)) for row in without_pyro(IGNORED_PARAMETERS)],
        ["package", "parameter", "why_ignored"],
    )

    if args.rules == "vignette":
        write_csv(
            out / "vignette_accepted_values.csv",
            [dict(zip(("package", "parameter", "accepted_value", "developer_source"), row)) for row in without_pyro(VIGNETTE_SOURCES)],
            ["package", "parameter", "accepted_value", "developer_source"],
        )
    write_readme(out / "README.md", overview, args.rules)

    for row in overview:
        print(
            f"{row['unit']:<10} {row['package_group']:<24} default={row['default']:>5} "
            f"non_default={row['non_default']:>5} unresolved={row['unresolved']:>5}  "
            f"default% resolved={row['default_pct_of_resolved']:>5}  range {row['default_pct_low']}-{row['default_pct_high']}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
