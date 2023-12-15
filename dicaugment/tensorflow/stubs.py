__all__ = ["ToTensorflow"]


class ToTensorflow:
    """
    Raises Error because tensorflow is not installed
    """

    def __init__(self, *args, **kwargs):
        cls_name = self.__class__.__name__
        raise ImportError(
            f"You are trying to use an augmentation '{cls_name}' that depends on the tensorflow library, "
            "but tensorflow is not installed.\n\n"
            "Either install tensorflow directly by running "
            "`pip install tensorflow`\n"
            "or by installing a version of DICaugment that contains tensorflow by running "
            "`pip install dicaugment[tensorflow]`"
        )
