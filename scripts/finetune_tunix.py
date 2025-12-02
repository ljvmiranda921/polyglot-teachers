import jax
import numpy as np
from flax import nnx  # neural network lib for jax
import jax.numpy as jnp  # numpy commands in TPU
from orbax import checkpoint as ocp  # checkpointing
import qwix  # quantization
import optax  # gradient and optimization library
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import params as gemma_params


def main():
    pass


if __name__ == "__main__":
    main()
