from .models import register, make, models
from . import transformer
from . import bottleneck
from . import loss
from . import larp_ar
from . import gptc
from . import larp_tokenizer
from . import vq
from .model_sem import auto_rae as autoencoder_sem
from .model_sem import auto_vqrae as autoencoder_sem_vae

def get_model_cls(name):
    return models[name]