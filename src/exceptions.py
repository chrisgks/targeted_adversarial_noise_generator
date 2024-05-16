class AdversarialMethodNotSupportedError(Exception):

    def __init__(self, message: str = "Selected method not supported!"):
        self.message: str = message
        super().__init__(self.message)


class ImageTypeNotSupportedError(Exception):

    def __init__(self, message: str = "Image type not supported!"):
        self.message: str = message
        super().__init__(self.message)
