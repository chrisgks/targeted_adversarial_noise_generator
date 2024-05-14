class AdversarialMethodNotSupportedError(Exception):

    def __init__(self, method: str, message: str = "Selected method not supported!"):
        self.method: str = method
        self.message: str = message
        super().__init__(self.message)
