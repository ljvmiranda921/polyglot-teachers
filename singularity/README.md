# Singularity Containers

## Available Definitions

- **`mtep.def`** - Base container (no extras)
- **`mtep_eval.def`** - With `--extra eval` for lighteval

## Build

```bash
singularity build --fakeroot mtep_eval.sif singularity/mtep_eval.def
```

## Usage

See main [README](../README.md#running-on-isambard-using-singularity) for usage instructions.
