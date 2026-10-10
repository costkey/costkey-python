"""Framework integrations that turn agent frameworks' hooks into CostKey spans.

Each integration imports its framework lazily, so installing CostKey never
pulls in an agent framework.
"""
