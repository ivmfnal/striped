class StripedNotFoundException(Exception):
    def __init__(self, message=""):
        self.Message = message
        
    def __str__(self):
        return "%s" % (self.Message,)
    
