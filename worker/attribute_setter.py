def attr_setter(method):
    def decorated(self, name, value):
        #print "decorated(%s, %s)" % (name, value)
        if self.__dict__.get("_in_constructor") or name[0] == '_':
            self.__dict__[name] = value
        else:
            return method(self, name, value)
    return decorated

def constructor(method):
    def decorated(self, *params, **args):
        self.__dict__["_in_constructor"] = True
        try:   
            method(self, *params, **args)
        finally:
            self.__dict__["_in_constructor"] = False
    return decorated

