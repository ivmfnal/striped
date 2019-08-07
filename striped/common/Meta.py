class MetaOperation:

    def __init__(self, op):
        self.Op = op
        
    def eval(self, profile, left, right):
        if isinstance(left, MetaNode):    left=left.eval(profile)
        if isinstance(right, MetaNode):    right=right.eval(profile)
        if self.Op == "<=": return left <= right
        elif self.Op == "<": return left < right
        elif self.Op == ">": return left > right
        elif self.Op == ">=": return left >= right
        elif self.Op == "==": return left == right
        elif self.Op == "!=": return left != right
        elif self.Op == "+": return left + right
        elif self.Op == "-": return left - right
        elif self.Op == "*": return left * right
        elif self.Op == "/": return left / right
        elif self.Op == "%": return left % right
        elif self.Op == "and": return left and right
        elif self.Op == "or": return left or right
        elif self.Op == "neg": 
            if isinstance(left, bool):
                return not left
            else:
                return -left
        elif self.Op == "__": return profile[left]
        else:
            raise ValueError("Unknown operation: %s" % (self.Op,))
        
class MetaNode:
    def __init__(self, op, left, right=None):
        self.Op = MetaOperation(op)
        self.Left = left
        self.Right = right
        
    def __le__(self, right):
        return MetaNode("<=", self, right)

    def __lt__(self, right):
        return MetaNode("<", self, right)

    def __ge__(self, right):
        return MetaNode(">=", self, right)

    def __gt__(self, right):
        return MetaNode(">", self, right)

    def __ne__(self, right):
        return MetaNode("!=", self, right)

    def __eq__(self, right):
        return MetaNode("==", self, right)
        
    def __add__(self, right):
        return MetaNode("+", self, right)
        
    def __mul__(self, right):
        return MetaNode("*", self, right)
        
    def __sub__(self, right):
        return MetaNode("-", self, right)
        
    def __div__(self, right):
        return MetaNode("/", self, right)
        
    def __mod__(self, right):
        return MetaNode("%", self, right)
        
    def __neg__(self):
        return MetaNode("neg", self)

    def __and__(self, right):
        return MetaNode("and", self, right)

    def __or__(self, right):
        return MetaNode("or", self, right)
        
    def __bool__(self):
        raise ValueError("""When building a frame selection expression, use '&', '|', '-' instead of boolean operators 'and', 'or', 'not' respectively. 
        Also keep in mind that '&', '|', '-' take presidence over comparison operators, unlike 'and', 'or', 'not'.
        """)
        
    def eval(self, profile):
        return self.Op.eval(profile, self.Left, self.Right)
        
    def serialize(self):
        left = self.Left.serialize() if isinstance(self.Left, MetaNode) else self.Left
        right = self.Right.serialize() if isinstance(self.Right, MetaNode) else self.Right
        return (self.Op.Op, left, right)

    @staticmethod        
    def deserialize(x):
        if isinstance(x, (tuple,list)):
            op = x[0]
            if op == "__":
                return Meta(x[1])
            else:
                return MetaNode(op, MetaNode.deserialize(x[1]), MetaNode.deserialize(x[2]))
        else:
            return x        
        
class Meta(MetaNode):

    def __init__(self, name):
        MetaNode.__init__(self, "__", name)
        
if __name__ == "__main__":
    
    profile = dict(x=1, y=2, z=-1)

    #exp = Meta("x")-1 < Meta("y")-1 & Meta("z") < 0
    exp = (Meta("x") < Meta("y")) & ( Meta("y") < 0 )
    print(exp.serialize())
    print(exp.eval(profile))
    
    exp = -exp
    
    print(exp.serialize())
    
    exp1 = MetaNode.deserialize(exp.serialize())
    print(exp1.serialize())
        
