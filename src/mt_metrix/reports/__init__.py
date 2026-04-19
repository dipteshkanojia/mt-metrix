"""Post-hoc reporting utilities (paper-matrix tables, etc.).

The runner writes one output directory per (domain, scorer-set) combination.
``tabulate`` walks a glob of those directories and produces the cross-domain,
cross-language correlation tables needed to extend the ALOPE paper.
"""
