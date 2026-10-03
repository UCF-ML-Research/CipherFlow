# ir/

## Purpose

Defines the Intermediate Representation (IR) of transformer computation.

## Responsibilities

- Operator schema definitions.
- Transformer graph representation.
- Backend-independent computation description.

## Expected Core Files

- `operator_schema.py`: canonical operator metadata schema.
- `bert_graph.py`: transformer graph representation using operator nodes.

## Rules

IR must not depend on specific backends.
