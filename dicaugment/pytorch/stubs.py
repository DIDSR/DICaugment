__all__ = ["ToPytorch"]


class ToPytorch:
    """
    Raises Error because torch is not installed
    """

    def __init__(self, *args, **kwargs):
        cls_name = self.__class__.__name__
        raise ImportError(
            f"You are trying to use an augmentation '{cls_name}' that depends on the torch library, "
            "but torch is not installed.\n\n"
            "Either install torch directly by running "
            "`pip install torch torchvision torchaudio`\n"
            "or by installing a version of DICaugment that contains torch by running "
            "`pip install dicaugment[torch]`"
        )
