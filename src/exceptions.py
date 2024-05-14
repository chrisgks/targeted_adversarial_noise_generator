class AdversarialMethodNotSupportedError(Exception):

    def __init__(self, method, message="Selected method not supported!"):
        self.method = method
        self.message = message
        super().__init__(self.message)
